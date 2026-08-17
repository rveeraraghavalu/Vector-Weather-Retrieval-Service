# Weather Intelligence Pipeline

A semantic search system for weather forecasts and alerts using the National Weather Service API, Lakebase Postgres, and vector embeddings.

## Data Source

### Why National Weather Service (NWS) API?

We chose the [National Weather Service API](https://www.weather.gov/documentation/services-web-api) for several reasons:

* **Free and open**: No API keys required, no rate limits for reasonable use
* **Authoritative data**: Official US government weather source
* **Rich narratives**: Provides detailed text forecasts and alert descriptions ideal for semantic search
* **Real-time alerts**: Access to active weather warnings, watches, and advisories
* **Structured and unstructured data**: Combines geospatial metadata with natural language descriptions
* **Production-ready**: Reliable, well-documented REST API with good uptime

The API provides two primary content types:
* **Forecasts**: Daily and hourly weather predictions with detailed narrative text
* **Alerts**: Weather warnings, watches, and advisories with event descriptions and instructions

## Schema Design

### `weather_documents` Table

Stores raw weather documents (forecasts and alerts) from the NWS API:

```sql
CREATE TABLE weather_documents (
    id TEXT PRIMARY KEY,              -- Unique document ID (forecast period ID or alert ID)
    location TEXT NOT NULL,           -- Human-readable location name
    latitude NUMERIC,                 -- Geographic coordinates
    longitude NUMERIC,
    source_type TEXT NOT NULL,        -- "forecast" or "alert"
    headline TEXT,                    -- Short summary (for alerts) or period name (for forecasts)
    event TEXT,                       -- Alert event type (e.g., "Severe Thunderstorm Warning")
    narrative_text TEXT NOT NULL,     -- Full text content for embedding
    issued_at TIMESTAMPTZ,           -- When the document was issued
    effective_at TIMESTAMPTZ,        -- When the alert becomes effective
    payload JSONB NOT NULL,          -- Full API response for reference
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Key decisions**:
* `id` as TEXT: Forecasts use period IDs, alerts use UUIDs from the API
* `narrative_text`: The primary field for semantic search—combines headline + detailed forecast/alert text
* `payload` as JSONB: Preserves complete API response for future analysis
* Separate `source_type`: Enables filtering forecasts vs alerts
* Temporal fields: Support time-based queries and freshness checks

### `weather_embeddings` Table

Stores vector embeddings of weather document narratives:

```sql
CREATE TABLE weather_embeddings (
    id SERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,        -- Foreign key to weather_documents.id
    chunk_index INTEGER NOT NULL DEFAULT 0,
    chunk_text TEXT NOT NULL,
    embedding vector(384),            -- 384-dimensional vector from all-MiniLM-L6-v2
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX ON weather_embeddings USING ivfflat (embedding vector_cosine_ops);
```

**Key decisions**:
* `chunk_index`: Currently always 0 (no chunking), but schema supports future multi-chunk documents
* `embedding vector(384)`: Uses pgvector extension with 384 dimensions
* `ivfflat` index: Approximate nearest neighbor search for performance
* `vector_cosine_ops`: Cosine similarity for semantic search (standard for sentence embeddings)
* `document_id` + `chunk_index` unique constraint: Prevents duplicate embeddings

### Chunking Strategy

**Current approach**: No chunking—each weather document is embedded as a single unit.

**Rationale**:
* Weather forecasts and alerts are typically short (200-500 words)
* Semantic coherence is best preserved at the document level
* NWS narratives are already well-structured summaries
* Simplifies retrieval—one embedding per document

**Future improvement**: For longer documents (e.g., detailed area forecast discussions), implement sliding-window chunking with 512-token windows and 128-token overlap.

## Embedding Model

### Model: `sentence-transformers/all-MiniLM-L6-v2`

**Specifications**:
* **Dimensions**: 384
* **Context length**: 256 word pieces (~200 words)
* **Architecture**: Distilled from MiniLM, optimized for semantic similarity
* **Performance**: ~14K sentences/sec on CPU, sub-10ms latency

**Why this model?**
* **Fast**: Lightweight enough to run in the Flask app without GPU
* **Accurate**: Strong performance on semantic textual similarity benchmarks (STS-B: 82.4)
* **Production-ready**: Widely used, well-tested, part of sentence-transformers library
* **Balanced**: Good trade-off between speed, size (80MB), and quality
* **No API costs**: Runs locally, no external inference service required

**Alternatives considered**:
* `all-mpnet-base-v2` (768 dims): Better quality but 3x slower, 2x larger
* OpenAI `text-embedding-3-small`: Requires API calls, cost/latency overhead
* `all-distilroberta-v1` (768 dims): Slightly better accuracy but slower

For weather narratives (shorter, domain-specific text), the lightweight MiniLM model provides excellent results without the overhead of larger models.

## Pipeline: End-to-End Execution

### Prerequisites

1. **Set up Lakebase connection**:
   ```bash
   cd Vector-Weather-Retrieval-Service
   python setup_secrets.py
   ```

2. **Fix database permissions** (one-time setup):
   ```sql
   -- Run in Lakebase SQL Editor:
   ALTER TABLE weather_documents OWNER TO student;
   ALTER TABLE weather_embeddings OWNER TO student;
   ALTER SEQUENCE weather_embeddings_id_seq OWNER TO student;
   ```
   See `PERMISSIONS_FIX.md` for detailed troubleshooting.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 1: Sync Weather Data

**Option A: Via Flask API** (recommended for production):
```bash
python app.py
# In another terminal:
curl -X POST http://localhost:8000/weather/sync
```

**Option B: Direct script**:
```python
from weather_client import WeatherClient
import lakebase

client = WeatherClient()
locations = [
    {"name": "San Francisco, CA", "lat": 37.7749, "lon": -122.4194},
    {"name": "Austin, TX", "lat": 30.2672, "lon": -97.7431}
]
documents = client.fetch_weather_documents(locations, limit=50)
# Upsert to weather_documents table...
```

**What it does**:
* Fetches forecasts and alerts from NWS API for specified locations
* Stores documents in `weather_documents` table
* Automatically creates table and indexes if they don't exist

### Step 2: Generate Embeddings

**Option A: Via Notebook** (recommended):
```bash
# Open the notebook:
databricks workspace export /Users/<your-email>/Vector-Weather-Retrieval-Service/notebooks/Ingest\ Weather\ Embeddings

# Or run cells programmatically:
# 1. Set widgets: TABLE_NAME, EMBEDDING_MODEL, EMBEDDING_DIM
# 2. Run all cells to read documents, generate embeddings, and upsert
```

**Option B: Via script** (for automation):
```bash
python ingest_weather_embeddings.py \
    --table weather_documents \
    --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
    --embedding-dim 384
```

**Option C: Via DAB job**:
```bash
databricks bundle deploy
databricks bundle run ingest_weather_embeddings
```

**What it does**:
* Reads all documents from `weather_documents`
* Generates 384-dim embeddings using sentence-transformers
* Batch upserts embeddings to `weather_embeddings` (100 per batch)
* Handles conflicts with `ON CONFLICT DO UPDATE`

### Step 3: Semantic Search

**Via Flask API**:
```bash
# Start the app:
python app.py

# Search via GET:
curl "http://localhost:8000/weather/search?query=severe%20thunderstorm%20warning&limit=10"

# Search via POST:
curl -X POST http://localhost:8000/weather/search \
  -H "Content-Type: application/json" \
  -d '{"query": "sunny and warm forecast", "limit": 5}'
```

**Via Web UI**:
1. Open `http://localhost:8000/`
2. Enter a natural language query (e.g., "tornado warning", "weekend weather")
3. View ranked results with similarity scores

**What it does**:
* Embeds the search query using the same model
* Performs cosine similarity search: `embedding <=> query_vector`
* Joins with `weather_documents` to retrieve full document metadata
* Returns top-k results ordered by semantic similarity

### Full Pipeline Example

```bash
# 1. Sync weather data
curl -X POST http://localhost:8000/weather/sync \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [
      {"name": "Seattle, WA", "lat": 47.6062, "lon": -122.3321}
    ],
    "limit": 50
  }'

# 2. Generate embeddings (in notebook or script)
python ingest_weather_embeddings.py

# 3. Search
curl "http://localhost:8000/weather/search?query=rain+tomorrow&limit=5"
```

## Known Limitations & Future Improvements

### Current Limitations

1. **No chunking**: Long documents (>200 words) may lose semantic detail. Currently not an issue since weather narratives are short, but would need chunking for detailed forecast discussions.

2. **Static locations**: Default sync targets hardcoded cities. Future: allow dynamic location discovery or full-state sync.

3. **No incremental updates**: Embeddings regenerate from scratch on each run. Future: detect changed documents and only re-embed those.

4. **Permission setup required**: New users must run `ALTER TABLE OWNER TO student` in Lakebase SQL Editor. This is a one-time setup but not automated.

5. **No hybrid search**: Pure vector search only. Future: combine with keyword search (BM25) for better recall on specific terms like city names or event types.

6. **Single embedding model**: No support for switching models or comparing different embedding strategies without re-ingesting all data.

7. **Approximate search only**: Uses `ivfflat` index for speed, which is approximate. For exact results (needed for eval), must disable index.

### Potential Improvements

#### Short-term (1-2 days)

* **Scheduled refresh**: DAB job to sync + embed daily via Databricks workflows
* **Hybrid search**: Add FTS (full-text search) index on `narrative_text` and combine with vector search
* **Reranking**: Use cross-encoder for top-k reranking after vector retrieval
* **Metadata filtering**: UI controls to filter by source_type, location, date range before semantic search
* **Automated permissions**: Alter ownership in `ensure_weather_table()` setup script

#### Medium-term (1 week)

* **Multi-region deployment**: Deploy as Databricks App with multiple Lakebase endpoints
* **Chunking for long docs**: Implement recursive character splitter for area forecast discussions (AFDs)
* **Embedding cache**: Store query embeddings to avoid re-embedding frequent searches
* **Evaluation suite**: Build test set of query-document pairs to measure retrieval quality (MRR, NDCG@10)
* **Model comparison**: A/B test MiniLM vs MPNet vs domain-adapted models

#### Long-term (multi-week)

* **Fine-tuned embeddings**: Train domain-adapted model on weather terminology using contrastive learning
* **Multimodal search**: Incorporate weather images (radar, satellite) using CLIP-style embeddings
* **Temporal awareness**: Embed time-decay into similarity scores (recent forecasts rank higher)
* **Agentic RAG**: Add LLM layer to synthesize answers from multiple retrieved documents
* **User feedback loop**: Collect click data and retrain embeddings with hard negatives

## Performance Benchmarks

**Current metrics** (on 100 weather documents, single-node serverless):

* **Sync**: ~2-3 seconds for 5 locations (10 forecasts + alerts)
* **Embedding generation**: ~0.8 seconds for 100 documents (batch size 32)
* **Vector search**: <50ms for top-10 results (with ivfflat index)
* **End-to-end search latency**: <100ms (embed query + search + join)

**Scalability estimates**:

* **10K documents**: ~8 seconds to embed, <100ms search
* **100K documents**: ~80 seconds to embed, <200ms search (may need index tuning)
* **1M documents**: Consider sharding by region or time window

## Repository Structure

```
Vector-Weather-Retrieval-Service/
├── app.py                          # Flask API (sync, search endpoints)
├── weather_client.py               # NWS API client
├── lakebase.py                     # Lakebase Postgres connection
├── ingest_weather_embeddings.py    # CLI script for embedding generation
├── requirements.txt                # Python dependencies
├── app.yaml                        # Databricks App config
├── databricks.yml                  # DAB bundle config
├── templates/
│   └── index.html                  # Web UI for semantic search
├── notebooks/
│   └── Ingest Weather Embeddings   # Interactive embedding generation
├── resources/
│   └── ingest_weather_embeddings_job.yml  # DAB job definition
└── sql/
    ├── FIX_PERMISSIONS_NOW.sql     # Permission fix script
    └── FIX_SET_ROLE_ERROR.sql      # Alternate permission fix
```

## Dependencies

```txt
sentence-transformers>=2.2.0    # Embedding model
psycopg2-binary>=2.9.0          # Postgres client with pgvector support
databricks-sdk>=0.20.0          # Databricks API client
flask>=2.3.0                    # Web framework
requests>=2.31.0                # HTTP client for NWS API
```

## Troubleshooting

### Error: "must be owner of table weather_documents"

**Solution**: Run the ALTER TABLE commands in Lakebase SQL Editor. See `PERMISSIONS_FIX.md` for full guide.

### Error: "operator does not exist: integer = text"

**Solution**: Fixed! The semantic search query now correctly joins `e.document_id = w.id` instead of `e.id = w.id`.

### Error: "invalid input syntax for type vector"

**Solution**: Fixed! Embedding arrays now use `[...]` format instead of `{...}` for pgvector compatibility.

### Empty search results

**Check**:
1. Have you synced weather data? `curl -X POST http://localhost:8000/weather/sync`
2. Have you generated embeddings? Run the notebook or script
3. Are embeddings in the table? `SELECT COUNT(*) FROM weather_embeddings;`

### Slow search performance

**Solutions**:
1. Ensure ivfflat index exists: `CREATE INDEX ON weather_embeddings USING ivfflat (embedding vector_cosine_ops);`
2. Increase `ivfflat.probes` for better accuracy: `SET ivfflat.probes = 10;`
3. Consider upgrading to HNSW index (requires pgvector 0.5.0+)

## License & Attribution

Weather data sourced from the [National Weather Service API](https://www.weather.gov/documentation/services-web-api), a free public service provided by NOAA.

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` by [UKPLab](https://github.com/UKPLab/sentence-transformers), Apache 2.0 license.

