"""
Weather Intelligence App:
- Serves a Flask API for weather data
- Fetches data from the National Weather Service API via weather_client.py
- Stores weather documents in Lakebase (Databricks-managed Postgres)
- Provides semantic search over weather forecasts and alerts using vector embeddings

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import json as _json
import logging
import os

import requests
from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request

import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-app")

app = Flask(__name__)
_w = WorkspaceClient()

# Weather document and embedding table names
WEATHER_TABLE_NAME = os.environ.get("WEATHER_TABLE_NAME", "weather_documents")
WEATHER_EMBEDDINGS_TABLE_NAME = os.environ.get("WEATHER_EMBEDDINGS_TABLE_NAME", "weather_embeddings")

# Default locations to fetch weather for
DEFAULT_LOCATIONS = [
    {"name": "San Francisco, CA", "lat": 37.7749, "lon": -122.4194},
    {"name": "New York, NY", "lat": 40.7128, "lon": -74.0060},
    {"name": "Chicago, IL", "lat": 41.8781, "lon": -87.6298},
    {"name": "Austin, TX", "lat": 30.2672, "lon": -97.7431},
    {"name": "Seattle, WA", "lat": 47.6062, "lon": -122.3321},
]

# Lazy-load the sentence transformer model for vector search
_embedding_model = None


def _get_embedding_model():
    """Lazy-load the sentence-transformers model for embedding generation."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
    return _embedding_model


def ensure_weather_table():
    """Create the weather documents table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            latitude NUMERIC,
            longitude NUMERIC,
            source_type TEXT NOT NULL,
            headline TEXT,
            event TEXT,
            narrative_text TEXT NOT NULL,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_location "
        f"ON {WEATHER_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_TABLE_NAME}_source_type "
        f"ON {WEATHER_TABLE_NAME} (source_type)"
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Weather intelligence dashboard with semantic search."""
    return render_template("index.html")


@app.route("/weather/documents")
def list_weather_documents():
    """List weather documents stored in Lakebase."""
    limit = int(request.args.get("limit", 100))
    location = request.args.get("location")
    source_type = request.args.get("source_type")  # "forecast" or "alert"
    
    query = f"SELECT * FROM {WEATHER_TABLE_NAME} WHERE 1=1"
    params = []
    
    if location:
        query += " AND location ILIKE %s"
        params.append(f"%{location}%")
    
    if source_type:
        query += " AND source_type = %s"
        params.append(source_type)
    
    query += " ORDER BY synced_at DESC LIMIT %s"
    params.append(limit)
    
    rows = lakebase.run_query(query, tuple(params))
    return jsonify(rows)


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """
    Fetch weather data (forecasts and alerts) from the National Weather Service
    and store it in Lakebase.
    
    Body (optional JSON): {"locations": [{"name": "...", "lat": X, "lon": Y}], "limit": 50}
    Defaults to DEFAULT_LOCATIONS when no locations are supplied.
    """
    ensure_weather_table()
    client = WeatherClient()
    
    body = request.json if request.is_json else {}
    locations = body.get("locations") or DEFAULT_LOCATIONS
    limit = int(body.get("limit", 50))
    
    # Fetch weather documents
    documents = client.fetch_weather_documents(locations, limit=limit)
    
    # Upsert into database
    total = _upsert_weather_batch(documents)
    
    return jsonify({"synced": total, "locations": len(locations)})


@app.route("/weather/forecast", methods=["GET"])
def get_forecast():
    """
    Get weather forecast for a specific location.
    
    Query params:
    - lat: Latitude (required)
    - lon: Longitude (required)
    - hourly: Return hourly forecast instead of daily (optional, default false)
    """
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "Valid lat and lon parameters are required"}), 400
    
    hourly = request.args.get("hourly", "").lower() == "true"
    
    client = WeatherClient()
    
    try:
        # Resolve grid point
        grid = client.get_grid_point(lat, lon)
        grid_id = grid.get("gridId")
        grid_x = grid.get("gridX")
        grid_y = grid.get("gridY")
        
        if not all([grid_id, grid_x, grid_y]):
            return jsonify({"error": "Could not resolve grid coordinates for location"}), 400
        
        # Fetch forecast
        if hourly:
            periods = client.get_hourly_forecast(grid_id, grid_x, grid_y)
        else:
            periods = client.get_forecast(grid_id, grid_x, grid_y)
        
        return jsonify({
            "lat": lat,
            "lon": lon,
            "grid_id": grid_id,
            "grid_x": grid_x,
            "grid_y": grid_y,
            "forecast_type": "hourly" if hourly else "daily",
            "periods": periods
        })
    
    except requests.HTTPError as e:
        logger.exception("Failed to fetch forecast")
        return jsonify({"error": f"Weather service error: {str(e)}"}), 500


@app.route("/weather/alerts", methods=["GET"])
def get_alerts():
    """
    Get active weather alerts.
    
    Query params:
    - state: Two-letter state code (optional, e.g. "TX", "CA")
    """
    state = request.args.get("state")
    
    client = WeatherClient()
    
    try:
        alerts = client.get_active_alerts(state=state)
        return jsonify({
            "state": state,
            "alert_count": len(alerts),
            "alerts": alerts
        })
    
    except requests.HTTPError as e:
        logger.exception("Failed to fetch alerts")
        return jsonify({"error": f"Weather service error: {str(e)}"}), 500


@app.route("/weather/search", methods=["GET", "POST"])
def search_weather_by_vector():
    """
    Perform semantic vector similarity search on weather documents.
    
    Accepts either GET with query params or POST with JSON body:
    - query: Natural language search query (required)
    - limit: Number of results to return (default: 10, max: 50)
    
    Example GET: /weather/search?query=severe thunderstorm warning&limit=10
    Example POST: {"query": "sunny and warm forecast", "limit": 20}
    """
    # Parse request parameters
    if request.method == "POST" and request.is_json:
        query = request.json.get("query", "")
        limit = int(request.json.get("limit", 10))
    else:
        query = request.args.get("query", "")
        limit = int(request.args.get("limit", 10))
    
    # Validate inputs
    query = query.strip() if isinstance(query, str) else ""
    
    if not query:
        return jsonify({"error": "Search query is required"}), 400
    
    # Clamp limit to reasonable range
    limit = max(1, min(limit, 50))
    
    try:
        # Generate embedding for the search query
        model = _get_embedding_model()
        query_embedding = model.encode([query])[0].tolist()
        
        # Format embedding as Postgres array literal for pgvector
        embedding_str = "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        
        # Query weather document embeddings
        documents = lakebase.run_query(
            f"""
            SELECT 
                w.id,
                w.location,
                w.source_type,
                w.headline,
                w.event,
                e.model_name,
                e.embedding <=> %s::vector AS distance,
                w.narrative_text,
                w.issued_at,
                w.effective_at,
                w.latitude,
                w.longitude
            FROM {WEATHER_EMBEDDINGS_TABLE_NAME} e
            LEFT JOIN {WEATHER_TABLE_NAME} w ON e.document_id = w.id
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding_str, embedding_str, limit),
        )
        
        return jsonify({
            "query": query,
            "documents": documents,
            "document_count": len(documents)
        })
        
    except ImportError:
        return jsonify({
            "error": "Vector search not available: sentence-transformers not installed"
        }), 500
    except Exception as e:
        logger.exception("Error performing vector search")
        return jsonify({"error": str(e)}), 500


def _upsert_weather_batch(documents: list[dict]) -> int:
    """Upsert a batch of weather documents into Lakebase."""
    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for doc in documents:
                cur.execute(
                    f"""
                    INSERT INTO {WEATHER_TABLE_NAME} (
                        id, location, latitude, longitude, source_type,
                        headline, event, narrative_text, issued_at, effective_at,
                        payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET location = EXCLUDED.location,
                            latitude = EXCLUDED.latitude,
                            longitude = EXCLUDED.longitude,
                            source_type = EXCLUDED.source_type,
                            headline = EXCLUDED.headline,
                            event = EXCLUDED.event,
                            narrative_text = EXCLUDED.narrative_text,
                            issued_at = EXCLUDED.issued_at,
                            effective_at = EXCLUDED.effective_at,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        doc.get("id"),
                        doc.get("location"),
                        doc.get("latitude"),
                        doc.get("longitude"),
                        doc.get("source_type"),
                        doc.get("headline"),
                        doc.get("event"),
                        doc.get("narrative_text"),
                        doc.get("issued_at"),
                        doc.get("effective_at"),
                        _json.dumps(doc.get("payload", {})),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")
