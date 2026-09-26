"""Grid Bricks — weather context provider.

`get_weather_context(site)` is the single entry point. Given a site dict from
the operational data warehouse (containing `city` and `state_province`), it
returns a compact forecast summary suitable for dropping into an LLM prompt,
or None if the site can't be geocoded or NWS can't provide a forecast.

Source: api.weather.gov (no API key required; identified via User-Agent).
Geocoding: offline lookup against the GeoNames cities dataset via the
`geonamescache` package, disambiguated by state.
"""
import logging
import os
from typing import Any, Optional

import geonamescache
import requests


logger = logging.getLogger(os.environ.get("DATABRICKS_APP_NAME", "grid-app"))

NWS_API_BASE = "https://api.weather.gov"
NWS_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT",
    "grid-app (https://www.databricks.com)",
)
REQUEST_TIMEOUT = 5
MAX_PERIODS = 8  # ~4 days of day/night periods — enough horizon for ops planning

# Build US-city indexes for offline geocoding (primary name + alternate names).
_gc = geonamescache.GeonamesCache()
_US_CITIES_BY_NAME: dict[str, list[dict[str, Any]]] = {}
_US_CITIES_BY_ALTNAME: dict[str, list[dict[str, Any]]] = {}
for _city in _gc.get_cities().values():
    if _city["countrycode"] != "US":
        continue
    _US_CITIES_BY_NAME.setdefault(_city["name"].lower(), []).append(_city)
    for _alt in _city.get("alternatenames", []):
        _US_CITIES_BY_ALTNAME.setdefault(_alt.lower(), []).append(_city)

session = requests.Session()
session.headers.update({
    "User-Agent": NWS_USER_AGENT,
    "Accept": "application/geo+json",
})


def geocode_site(site: dict) -> Optional[dict[str, Any]]:
    """Resolve a site dict to a US city record with lat/lon.

    Uses `state_province` to disambiguate same-named cities (Springfield, MO
    vs Springfield, MA). Falls back to highest-population match if the state
    filter yields nothing.
    """
    city_name = (site.get("city") or "").strip()
    state = (site.get("state_province") or "").strip().upper()

    if not city_name:
        return None

    key = city_name.lower()
    matches = _US_CITIES_BY_NAME.get(key) or _US_CITIES_BY_ALTNAME.get(key) or []
    if not matches:
        return None

    if state:
        state_matches = [c for c in matches if c.get("admin1code") == state]
        if state_matches:
            matches = state_matches

    best = max(matches, key=lambda c: c["population"])
    return {
        "name":  best["name"],
        "state": best.get("admin1code", ""),
        "lat":   best["latitude"],
        "lon":   best["longitude"],
    }


def fetch_forecast(lat: float, lon: float) -> list[dict[str, Any]]:
    """Two-hop NWS lookup: /points → forecast URL → list of forecast periods."""
    response = session.get(f"{NWS_API_BASE}/points/{lat},{lon}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    forecast_url = response.json()["properties"]["forecast"]

    response = session.get(forecast_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json().get("properties", {}).get("periods", [])


def _format_wind(period: dict[str, Any]) -> Optional[str]:
    speed = period.get("windSpeed")
    direction = period.get("windDirection")
    if speed and direction:
        return f"{speed} {direction}"
    return speed or direction or None


def summarize_forecast(periods: list[dict[str, Any]], resolved: dict[str, Any]) -> dict[str, Any]:
    """Trim NWS's forecast periods to a compact LLM-friendly summary.

    NWS already gives us natural period names ("Today", "Tonight", "Tuesday
    Night"), so we don't need to compute any relative labels ourselves —
    we just pass them through.
    """
    trimmed = periods[:MAX_PERIODS]

    return {
        "location":        f"{resolved['name']}, {resolved['state']}".strip(", "),
        "source":          "National Weather Service",
        "forecast_window": f"Next {len(trimmed)} periods (NWS day/night)",
        "periods": [
            {
                "name":                 p.get("name"),
                "is_daytime":           p.get("isDaytime"),
                "temperature":          p.get("temperature"),
                "temperature_unit":     f"°{p.get('temperatureUnit', 'F')}",
                "wind":                 _format_wind(p),
                "precipitation_chance": (p.get("probabilityOfPrecipitation") or {}).get("value"),
                "short_forecast":       p.get("shortForecast"),
            }
            for p in trimmed
        ],
    }


def get_weather_context(site: dict) -> Optional[dict[str, Any]]:
    resolved = geocode_site(site)
    if not resolved:
        logger.warning(
            "Could not geocode site for weather context: site_name=%r city=%r state=%r",
            site.get("site_name"),
            site.get("city"),
            site.get("state_province"),
        )
        return None

    try:
        periods = fetch_forecast(resolved["lat"], resolved["lon"])
    except requests.RequestException as exc:
        logger.warning(
            "NWS forecast fetch failed for site_name=%r location=%r error=%r",
            site.get("site_name"),
            resolved["name"],
            exc,
        )
        return None
    except (KeyError, ValueError) as exc:
        logger.warning(
            "NWS forecast response malformed for site_name=%r location=%r error=%r",
            site.get("site_name"),
            resolved["name"],
            exc,
        )
        return None

    if not periods:
        logger.warning("NWS returned no forecast periods for site_name=%r", site.get("site_name"))
        return None

    return summarize_forecast(periods, resolved)
