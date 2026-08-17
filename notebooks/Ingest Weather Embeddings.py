# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Reads weather documents from the `weather_documents` table in Lakebase
# MAGIC    (forecasts and alerts synced by the Weather Intelligence Flask app)
# MAGIC 2. Computes sentence embeddings for each document's narrative text using
# MAGIC    sentence-transformers
# MAGIC 3. Upserts the embeddings into the `weather_embeddings` table with pgvector
# MAGIC    for semantic search
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Weather documents must be synced first (via Flask app's `/weather/sync` endpoint)
# MAGIC - Tables must exist: `weather_documents`, `weather_embeddings`
# MAGIC - pgvector extension must be enabled in Lakebase

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override table names and the embedding model without editing
# MAGIC the notebook - useful when running this as a scheduled Databricks Job.

# COMMAND ----------

# DBTITLE 1,Setup widgets for configuration
dbutils.widgets.text("documents_table_name", "weather_documents", "Source table (weather docs)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("embedding_dim", "384", "Embedding dimension (384 for all-MiniLM-L6-v2)")

# Read widget values
DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
EMBEDDING_DIM = int(dbutils.widgets.get("embedding_dim"))

print(f"Source: {DOCUMENTS_TABLE_NAME}")
print(f"Destination: {EMBEDDINGS_TABLE_NAME}")
print(f"Model: {EMBEDDING_MODEL_NAME}")
print(f"Dimension: {EMBEDDING_DIM}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope.

# COMMAND ----------

# DBTITLE 1,Parse Lakebase connection info
import base64
from urllib.parse import urlparse
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")

lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connected to: {db_host}:{db_port}/{db_name}")
print(f"User: {db_user}")

# COMMAND ----------

# DBTITLE 1,Test psycopg2 connection
import psycopg2

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using role: {db_user}\n")

try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
        connect_timeout=10
    )
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"✅ Connection successful!")
    print(f"PostgreSQL version: {version[:50]}...")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load weather documents
# MAGIC
# MAGIC Reads all documents from the `weather_documents` table that don't already
# MAGIC have embeddings. Each document's `narrative_text` field will be embedded.

# COMMAND ----------

# DBTITLE 1,Load weather documents from Lakebase
import pandas as pd
import psycopg2

print(f"Loading weather documents from {DOCUMENTS_TABLE_NAME}...")

# Connect and load documents
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    # Load documents that don't have embeddings yet
    query = f"""
        SELECT 
            d.id,
            d.location,
            d.latitude,
            d.longitude,
            d.source_type,
            d.headline,
            d.event,
            d.narrative_text,
            d.issued_at
        FROM {DOCUMENTS_TABLE_NAME} d
        LEFT JOIN {EMBEDDINGS_TABLE_NAME} e ON d.id = e.document_id
        WHERE e.document_id IS NULL
        ORDER BY d.synced_at DESC;
    """
    
    weather_df = pd.read_sql(query, conn)
    print(f"Loaded {len(weather_df)} weather documents without embeddings")
    
    if len(weather_df) == 0:
        print("\n✅ All documents already have embeddings! Nothing to process.")
    else:
        print(f"\nSample documents:")
        display(weather_df.head(3))
        
finally:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings
# MAGIC
# MAGIC Loads the sentence-transformers model and computes embeddings for the
# MAGIC `narrative_text` field of each weather document in batches.

# COMMAND ----------

# DBTITLE 1,Compute embeddings using sentence-transformers
import os
from sentence_transformers import SentenceTransformer

if len(weather_df) == 0:
    print("No documents to embed. Skipping.")
else:
    # Set up HuggingFace cache
    os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"
    
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")
    
    # Compute embeddings in batches
    print(f"Computing embeddings for {len(weather_df)} documents...")
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(weather_df), batch_size):
        batch = weather_df.iloc[i:i+batch_size]
        # Embed the narrative_text field
        vectors = model.encode(batch["narrative_text"].tolist(), show_progress_bar=False)
        all_embeddings.extend(vectors.tolist())
        if (i + batch_size) % 128 == 0:
            print(f"  Processed {min(i + batch_size, len(weather_df))}/{len(weather_df)} documents")
    
    # Add embeddings to dataframe
    weather_df["embedding"] = all_embeddings
    
    print(f"\n✅ Computed {len(weather_df)} embeddings using {EMBEDDING_MODEL_NAME}")
    print(f"Embedding dimension: {len(all_embeddings[0])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Inserts the computed embeddings into the `weather_embeddings` table.
# MAGIC Each embedding is stored as a pgvector type for efficient similarity search.

# COMMAND ----------

# DBTITLE 1,Insert embeddings using psycopg2
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

if len(weather_df) == 0:
    print("No embeddings to insert.")
else:
    print(f"Inserting {len(weather_df)} embeddings into {EMBEDDINGS_TABLE_NAME}...")
    
    # Connect to Lakebase
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Prepare data for batch insert
        # Format: (document_id, chunk_index, chunk_text, embedding, model_name, created_at)
        insert_data = []
        for _, row in weather_df.iterrows():
            # Format embedding as PostgreSQL array literal: '[val1,val2,...]'
            embedding_str = '[' + ','.join(str(float(x)) for x in row['embedding']) + ']'
            
            insert_data.append((
                row['id'],                    # document_id
                0,                            # chunk_index (always 0 for full docs)
                row['narrative_text'],        # chunk_text
                embedding_str,                # embedding as array string
                EMBEDDING_MODEL_NAME,         # model_name
                datetime.now()                # created_at
            ))
        
        # Batch insert with ON CONFLICT handling
        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                document_id, chunk_index, chunk_text, embedding, model_name, created_at
            ) VALUES %s
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name,
                created_at = EXCLUDED.created_at
        """
        
        # execute_values for efficient batch insert
        template = "(%s, %s, %s, %s::vector, %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        inserted_count = cursor.rowcount
        
        print(f"\n✅ Successfully inserted/updated {inserted_count} embeddings")
        print(f"Model: {EMBEDDING_MODEL_NAME}")
        print(f"Dimension: {EMBEDDING_DIM}")
        
    except Exception as e:
        print(f"\n❌ Error inserting embeddings: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Query the embeddings table to verify the results.

# COMMAND ----------

# DBTITLE 1,Verify embeddings in database
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    # Count embeddings by model
    count_query = f"""
        SELECT 
            model_name,
            COUNT(*) as embedding_count,
            COUNT(DISTINCT document_id) as unique_documents
        FROM {EMBEDDINGS_TABLE_NAME}
        GROUP BY model_name
        ORDER BY embedding_count DESC;
    """
    
    count_df = pd.read_sql(count_query, conn)
    print("Embeddings by model:")
    display(count_df)
    
    # Sample recent embeddings
    sample_query = f"""
        SELECT 
            e.document_id,
            d.location,
            d.source_type,
            d.headline,
            e.model_name,
            e.created_at
        FROM {EMBEDDINGS_TABLE_NAME} e
        JOIN {DOCUMENTS_TABLE_NAME} d ON e.document_id = d.id
        ORDER BY e.created_at DESC
        LIMIT 5;
    """
    
    sample_df = pd.read_sql(sample_query, conn)
    print("\nRecent embeddings:")
    display(sample_df)
    
finally:
    conn.close()

print("\n✅ Weather embeddings pipeline complete!")

# COMMAND ----------

