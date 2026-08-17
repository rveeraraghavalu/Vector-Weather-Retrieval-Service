#!/usr/bin/env python
"""
Weather Document Embedding Pipeline

Reads unembedded weather documents from weather_documents table,
chunks narrative text, generates embeddings using sentence-transformers,
and writes embeddings to weather_embeddings table.

Usage:
    python ingest_weather_embeddings.py

Requires:
    - sentence-transformers
    - psycopg2
    - databricks-sdk (for secrets)
"""

import logging
import os
from typing import List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from sentence_transformers import SentenceTransformer

import lakebase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
WEATHER_DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE_NAME", "weather_documents")
WEATHER_EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")

# Embedding model - same as news pipeline for compatibility
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Chunking parameters
CHUNK_SIZE = 800  # Characters per chunk
CHUNK_OVERLAP = 100  # Overlap between chunks
BATCH_SIZE = 100  # Embeddings to write per batch


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Most NWS text is short enough that chunking may only matter for
    combined alert+instruction text, but we apply it uniformly.
    """
    if not text or len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        
        # Move start forward, accounting for overlap
        start = end - overlap
    
    return chunks


def get_unembedded_documents() -> List[dict]:
    """
    Fetch weather documents that haven't been embedded yet.
    """
    query = f"""
        SELECT d.id, d.location, d.headline, d.narrative_text
        FROM {WEATHER_DOCUMENTS_TABLE} d
        LEFT JOIN {WEATHER_EMBEDDINGS_TABLE} e ON d.id = e.document_id
        WHERE e.document_id IS NULL
        ORDER BY d.synced_at DESC
    """
    
    rows = lakebase.run_query(query)
    logger.info(f"Found {len(rows)} unembedded documents")
    return rows


def generate_embeddings(
    documents: List[dict],
    model: SentenceTransformer
) -> List[Tuple[str, int, str, List[float]]]:
    """
    Generate embeddings for document chunks.
    
    Returns:
        List of (document_id, chunk_index, chunk_text, embedding_vector)
    """
    embedding_records = []
    
    for doc in documents:
        doc_id = doc["id"]
        narrative = doc.get("narrative_text", "")
        
        if not narrative:
            logger.warning(f"Document {doc_id} has no narrative text, skipping")
            continue
        
        # Chunk the narrative text
        chunks = chunk_text(narrative)
        
        # Generate embeddings for all chunks at once (batched for efficiency)
        if chunks:
            embeddings = model.encode(chunks, show_progress_bar=False)
            
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Convert numpy array to Python list
                embedding_list = embedding.tolist()
                embedding_records.append((doc_id, idx, chunk, embedding_list))
    
    logger.info(f"Generated {len(embedding_records)} embeddings from {len(documents)} documents")
    return embedding_records


def write_embeddings_batch(embeddings: List[Tuple[str, int, str, List[float]]], model_name: str):
    """
    Write a batch of embeddings to the weather_embeddings table using psycopg2.
    
    Uses execute_values for efficient bulk insertion.
    """
    if not embeddings:
        return 0
    
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            # Prepare data for execute_values
            # Format: (document_id, chunk_index, chunk_text, embedding_str, model_name)
            values = [
                (
                    doc_id,
                    chunk_idx,
                    chunk_text,
                    # Format embedding as array string for PostgreSQL
                    "[" + ",".join(str(float(x)) for x in embedding) + "]",
                    model_name
                )
                for doc_id, chunk_idx, chunk_text, embedding in embeddings
            ]
            
            # Use execute_values for bulk insert
            execute_values(
                cur,
                f"""
                INSERT INTO {WEATHER_EMBEDDINGS_TABLE} 
                (document_id, chunk_index, chunk_text, embedding, model_name, created_at)
                VALUES %s
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        created_at = EXCLUDED.created_at
                """,
                values,
                template="(%s, %s, %s, %s::vector, %s, now())"
            )
            
            conn.commit()
    
    return len(embeddings)


def main():
    """
    Main pipeline execution.
    """
    logger.info("Starting weather embedding pipeline")
    logger.info(f"Model: {EMBEDDING_MODEL}")
    logger.info(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    
    # Load embedding model
    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded successfully")
    
    # Fetch unembedded documents
    documents = get_unembedded_documents()
    
    if not documents:
        logger.info("No unembedded documents found. Exiting.")
        return
    
    # Process in batches
    total_written = 0
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} documents)")
        
        # Generate embeddings
        embeddings = generate_embeddings(batch, model)
        
        # Write to database
        written = write_embeddings_batch(embeddings, EMBEDDING_MODEL)
        total_written += written
        logger.info(f"Wrote {written} embeddings to database")
    
    logger.info(f"Pipeline complete. Total embeddings written: {total_written}")


if __name__ == "__main__":
    main()
