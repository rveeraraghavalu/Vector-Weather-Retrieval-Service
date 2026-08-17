"""
Client for the National Weather Service (NWS) API.

No API key required - free public API with generous rate limits.
Docs: https://www.weather.gov/documentation/services-web-api
"""

import hashlib
import logging
from datetime import datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.weather.gov"
_DEFAULT_TIMEOUT = 30


class WeatherClient:
    """Thin wrapper around the National Weather Service API."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # NWS API requires a User-Agent header
        self._session.headers.update(
            {
                "User-Agent": "(Databricks Weather Retrieval Service, contact@example.com)",
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a GET request against the NWS API."""
        resp = self._session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def get_grid_point(self, lat: float, lon: float) -> dict:
        """
        Resolve a lat/lon to NWS grid coordinates.
        
        Returns: {
            "properties": {
                "gridId": "TOP",
                "gridX": 31,
                "gridY": 80,
                "forecast": "https://api.weather.gov/gridpoints/TOP/31,80/forecast",
                "forecastHourly": "...",
                ...
            }
        }
        """
        data = self.get(f"/points/{lat},{lon}")
        return data.get("properties", {})

    def get_forecast(self, grid_id: str, grid_x: int, grid_y: int) -> list[dict]:
        """
        Fetch the multi-day forecast for a grid point.
        
        Each period has:
        - number, name (e.g. "Tonight", "Wednesday")
        - startTime, endTime
        - temperature, temperatureUnit
        - windSpeed, windDirection
        - shortForecast, detailedForecast (narrative text)
        """
        data = self.get(f"/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast")
        return data.get("properties", {}).get("periods", [])

    def get_hourly_forecast(
        self, grid_id: str, grid_x: int, grid_y: int
    ) -> list[dict]:
        """
        Fetch the hourly forecast for a grid point.
        
        Each period has hourly granularity with similar fields as daily forecast.
        """
        data = self.get(f"/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast/hourly")
        return data.get("properties", {}).get("periods", [])

    def get_active_alerts(self, state: str | None = None, area: str | None = None) -> list[dict]:
        """
        Fetch active weather alerts.
        
        Args:
            state: Two-letter state code (e.g. "TX", "CA")
            area: Area code (use state for simplicity)
        
        Each alert has:
        - id (unique alert ID)
        - event (e.g. "Flash Flood Warning")
        - headline
        - description (free-text narrative)
        - instruction (what to do)
        - severity, urgency, certainty
        - effective, expires
        - areaDesc
        """
        params = {}
        if state:
            params["area"] = state.upper()
        elif area:
            params["area"] = area.upper()
        
        data = self.get("/alerts/active", params=params)
        features = data.get("features", [])
        return [f.get("properties", {}) for f in features]

    def fetch_weather_documents(
        self, locations: list[dict], limit: int = 50
    ) -> list[dict]:
        """
        Fetch weather documents (alerts + forecasts) for a list of locations.
        
        Args:
            locations: List of {"lat": float, "lon": float, "name": str}
            limit: Max forecast periods per location
        
        Returns:
            List of normalized document dicts ready for DB insertion.
        """
        documents = []
        
        for loc in locations:
            lat = loc.get("lat")
            lon = loc.get("lon")
            location_name = loc.get("name", f"{lat},{lon}")
            
            if lat is None or lon is None:
                logger.warning(f"Skipping location {location_name}: missing lat/lon")
                continue
            
            try:
                # Resolve grid point
                grid = self.get_grid_point(lat, lon)
                grid_id = grid.get("gridId")
                grid_x = grid.get("gridX")
                grid_y = grid.get("gridY")
                
                if not all([grid_id, grid_x, grid_y]):
                    logger.warning(
                        f"Could not resolve grid for {location_name} ({lat},{lon})"
                    )
                    continue
                
                # Fetch forecast
                try:
                    forecast_periods = self.get_forecast(grid_id, grid_x, grid_y)
                    for period in forecast_periods[:limit]:
                        doc = self._normalize_forecast(period, location_name, lat, lon)
                        if doc:
                            documents.append(doc)
                except requests.HTTPError as e:
                    logger.warning(f"Failed to fetch forecast for {location_name}: {e}")
                
                # Fetch active alerts for the state/area
                # Extract state from location name if possible (e.g. "Chicago, IL")
                state = None
                if "," in location_name:
                    parts = location_name.split(",")
                    if len(parts) >= 2:
                        state = parts[-1].strip()
                
                if state and len(state) == 2:
                    try:
                        alerts = self.get_active_alerts(state=state)
                        for alert in alerts:
                            doc = self._normalize_alert(alert, location_name, lat, lon)
                            if doc:
                                documents.append(doc)
                    except requests.HTTPError as e:
                        logger.warning(f"Failed to fetch alerts for {location_name}: {e}")
                
            except requests.HTTPError as e:
                logger.warning(f"Error fetching weather for {location_name}: {e}")
                continue
        
        return documents

    def _normalize_forecast(self, period: dict, location: str, lat: float, lon: float) -> dict | None:
        """
        Normalize a forecast period into a document record.
        """
        detailed = period.get("detailedForecast", "")
        short = period.get("shortForecast", "")
        
        if not detailed and not short:
            return None
        
        # Combine short and detailed forecast as narrative text
        narrative = f"{short}. {detailed}" if short and detailed else (detailed or short)
        
        start_time = period.get("startTime")
        name = period.get("name", "")
        number = period.get("number", 0)
        
        # Generate stable ID from location + start_time + number
        id_str = f"forecast_{location}_{start_time}_{number}"
        doc_id = hashlib.md5(id_str.encode()).hexdigest()
        
        return {
            "id": doc_id,
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "source_type": "forecast",
            "headline": name,
            "event": "Forecast",
            "narrative_text": narrative,
            "issued_at": start_time,
            "effective_at": start_time,
            "payload": period,
            "synced_at": datetime.utcnow().isoformat(),
        }

    def _normalize_alert(self, alert: dict, location: str, lat: float, lon: float) -> dict | None:
        """
        Normalize a weather alert into a document record.
        """
        alert_id = alert.get("id")
        if not alert_id:
            return None
        
        event = alert.get("event", "")
        headline = alert.get("headline", "")
        description = alert.get("description", "")
        instruction = alert.get("instruction", "")
        
        # Combine description + instruction as narrative text
        narrative_parts = []
        if description:
            narrative_parts.append(description)
        if instruction:
            narrative_parts.append(f"Instructions: {instruction}")
        
        narrative = " ".join(narrative_parts)
        
        if not narrative:
            return None
        
        return {
            "id": alert_id,
            "location": location,
            "latitude": lat,
            "longitude": lon,
            "source_type": "alert",
            "headline": headline or event,
            "event": event,
            "narrative_text": narrative,
            "issued_at": alert.get("sent"),
            "effective_at": alert.get("effective"),
            "payload": alert,
            "synced_at": datetime.utcnow().isoformat(),
        }
