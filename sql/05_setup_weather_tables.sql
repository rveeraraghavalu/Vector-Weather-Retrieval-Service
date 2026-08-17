-- Weather Documents Table
-- Stores raw weather data (alerts + forecasts) from National Weather Service API

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    latitude NUMERIC,
    longitude NUMERIC,
    source_type TEXT NOT NULL,  -- 'alert' or 'forecast'
    headline TEXT,
    event TEXT,
    narrative_text TEXT NOT NULL,  -- The text to embed
    issued_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    payload JSONB NOT NULL,  -- Raw API response for provenance
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location ON weather_documents (location);
CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type ON weather_documents (source_type);
CREATE INDEX IF NOT EXISTS idx_weather_documents_issued_at ON weather_documents (issued_at);

-- Weather Embeddings Table
-- Stores vector embeddings for semantic search over weather documents

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id SERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,  -- 384-dim for all-MiniLM-L6-v2
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Create HNSW index for efficient vector similarity search
-- Using cosine distance operator for semantic similarity
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding_hnsw
ON weather_embeddings
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id ON weather_embeddings (document_id);

-- Comments for documentation
COMMENT ON TABLE weather_documents IS 'Raw weather documents (alerts + forecasts) from National Weather Service API';
COMMENT ON TABLE weather_embeddings IS 'Vector embeddings for semantic search over weather narrative text';
COMMENT ON COLUMN weather_documents.narrative_text IS 'Free-text weather description to embed (detailedForecast or alert description+instruction)';
COMMENT ON COLUMN weather_embeddings.embedding IS '384-dimensional embedding from sentence-transformers/all-MiniLM-L6-v2';
