# Weather Intelligence Platform

A Databricks App that provides AI-powered weather intelligence through semantic search:
- Connects to **Lakebase** (Databricks-managed Postgres) for weather data storage
- Fetches weather forecasts and alerts from the **National Weather Service API** (free, no API key required)
- Vectorizes weather text and stores embeddings in Lakebase with pgvector
- Provides **semantic search** over weather documents via vector similarity
- Exposes a Flask REST API for weather forecasts, alerts, and searches
- Interactive web UI for exploring weather data

## Features

- **Real-time Weather Data**: Fetch current forecasts and active alerts from the National Weather Service
- **Semantic Search**: Natural language queries like "severe thunderstorm warnings" or "sunny weekend forecast"
- **Vector Embeddings**: Uses sentence-transformers to encode weather narratives for similarity search
- **Location-Based**: Get forecasts for any lat/lon coordinates
- **Alert Monitoring**: Track active weather alerts by state or nationwide
- **Postgres with pgvector**: Leverages Lakebase and pgvector for efficient vector storage and retrieval

## Files

- `app.py` - Flask app: weather API endpoints for forecasts, alerts, sync, and search
- `weather_client.py` - National Weather Service API client
- `lakebase.py` - Lakebase connection helper (psycopg2 + SQLAlchemy)
- `templates/index.html` - Weather intelligence dashboard UI
- `ingest_weather_embeddings.py` - ETL pipeline for generating weather embeddings
- `app.yaml` - Databricks App deployment config
- `.env.example` - Local dev env var template

## Step-by-step setup

### 1. Create a Lakebase instance

1. In your Databricks workspace, go to **Catalog** → **Lakebase** tab
2. Click **Create Lakebase instance**
   - Name it (e.g. `weather-intel-db`)
   - Choose appropriate capacity/region
   - Click **Create** and wait for **Available** status
3. Open the instance → **Roles & Databases** tab
4. **Enable native (password) authentication**
5. **Create a new role**:
   - Click **Add role** → **Password** authentication
   - Name the role (e.g. `weather_app`)
   - Copy the connection URL:
     ```
     postgresql://<role>:<password>@<host>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require
     ```

### 2. Store your Lakebase URL secret

Run from a **Databricks notebook**:

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
w.secrets.create_scope(scope="database")
w.secrets.put_secret(
    scope="database",
    key="lakebase-url",
    string_value="<your-lakebase-connection-url>"
)
```

### 3. Configure environment variables (local dev)

Copy `.env.example` to `.env` and add your Lakebase URL:

```bash
cp .env.example .env
# Edit .env and set LAKEBASE_URL=<your-connection-url>
```

For deployment, `app.yaml` pulls `LAKEBASE_URL` from the `database/lakebase-url` secret automatically.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run locally

```bash
python app.py
```

Visit http://localhost:8000 to access the weather dashboard.

### 6. Deploy to Databricks Apps

1. **Create a Git folder** in your Databricks workspace:
   - Sidebar → **Workspace** → **Create** → **Git folder**
   - Paste your Git repository URL
   - Click **Create Git folder**

2. **Create the Databricks App**:
   - Sidebar → **Compute** → **Apps**
   - Click **Create app** → **Custom**
   - Name it (e.g. `weather-intelligence`)
   - Select your Git folder as the source
   - Click **Deploy**

## API Endpoints

### Weather Data Sync
```
POST /weather/sync
Body (optional): {
  "locations": [
    {"name": "San Francisco, CA", "lat": 37.7749, "lon": -122.4194}
  ],
  "limit": 50
}
```
Fetches weather forecasts and alerts from NWS API and stores them in Lakebase.

### Get Forecast
```
GET /weather/forecast?lat=37.7749&lon=-122.4194&hourly=false
```
Returns weather forecast for the specified coordinates (daily or hourly).

### Get Active Alerts
```
GET /weather/alerts?state=CA
```
Returns active weather alerts for a state (or nationwide if state is omitted).

### Semantic Search
```
POST /weather/search
Body: {
  "query": "severe thunderstorm warnings",
  "limit": 10
}
```
Performs vector similarity search over weather documents.

### List Documents
```
GET /weather/documents?location=San Francisco&source_type=alert&limit=100
```
Returns stored weather documents filtered by location and/or source type.

## Running the Weather Embedding Pipeline

The `ingest_weather_embeddings.py` script is a self-contained ETL pipeline that:
1. Fetches weather alerts and forecasts from the National Weather Service API
2. Chunks weather narrative text into manageable pieces
3. Generates embeddings using `sentence-transformers/all-MiniLM-L6-v2`
4. Upserts documents and embeddings into Lakebase with pgvector

Run it manually:

```bash
python ingest_weather_embeddings.py
```

Or schedule it as a Databricks Workflow:
1. Upload `ingest_weather_embeddings.py` to your workspace
2. Create a new Job in Databricks Workflows
3. Add a Python task pointing to the script
4. Set a schedule (e.g., daily at 6:00 AM UTC)
5. Configure notifications for failures

The pipeline automatically:
- Creates necessary tables (`weather_documents` and `weather_embeddings`)
- Enables pgvector extension if not already enabled
- Fetches latest weather data from NWS API
- Stores processed documents and vectors for semantic search

## Web UI Features

The weather intelligence dashboard (`/`) provides:

1. **Sync Weather Data**: One-click fetch of latest forecasts and alerts
2. **Get Forecast**: Interactive map with pre-configured major cities or custom lat/lon
3. **Active Alerts**: Real-time weather alerts filtered by state
4. **Semantic Search**: Natural language queries over weather documents with relevance scoring

## Architecture

```
┌─────────────────┐
│   Flask Web UI  │
│  (index.html)   │
└────────┬────────┘
         │
         v
┌─────────────────┐      ┌──────────────────┐
│   Flask API     │─────>│  Weather Client  │
│    (app.py)     │      │ (NWS API calls)  │
└────────┬────────┘      └──────────────────┘
         │
         v
┌─────────────────┐      ┌──────────────────┐
│   Lakebase      │<─────│ Embedding ETL    │
│  (Postgres +    │      │  (sentence-      │
│   pgvector)     │      │   transformers)  │
└─────────────────┘      └──────────────────┘
```

## Technologies

- **Backend**: Flask, Python
- **Database**: Lakebase (Databricks-managed Postgres)
- **Vector Search**: pgvector extension
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Weather Data**: National Weather Service API (free, no auth required)
- **Frontend**: Vanilla JavaScript, responsive CSS

## License

This is a demo application showcasing Databricks capabilities.
