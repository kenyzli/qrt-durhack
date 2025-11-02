from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
import math
import numpy as np

from main import evaluate_naive_atOffice, OFFICES

app = Flask(__name__)

# Basic mapping from city names to IATA codes used by the existing functions.
# Extend this mapping as needed for your dataset.

CITY_TO_IATA = {
    "Mumbai": "BOM",
    "Shanghai": "PVG",
    "Hong Kong": "HKG",
    "Singapore": "SIN",
    "Sydney": "SYD",
    "London": "LHR",
    "New York": "JFK",
    "Paris": "CDG",
    "Dubai": "DXB",
    "Zurich": "ZRH",
    "Geneva": "GVA",
    "Aarhus": "AAR",
    "Wroclaw": "WRO",
    "Budapest": "BUD",
}

IATA_TO_CITY = {v: k for k, v in CITY_TO_IATA.items()}


def parse_iso_z(dt_str: str) -> datetime:
    # support trailing Z (UTC) and offsets
    if dt_str is None:
        return None
    s = dt_str
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    # Python 3.11+ supports fromisoformat with offsets; this will work on common ISO formats
    return datetime.fromisoformat(s)


@app.route("/plan", methods=["POST"])
def plan():
    payload = request.get_json()
    if not payload:
        return jsonify({"error": "invalid json"}), 400

    attendees = payload.get("attendees", {})
    window = payload.get("availability_window", {})
    duration = payload.get("event_duration", {})

    # map city names to IATA
    unknown = [c for c in attendees.keys() if c not in CITY_TO_IATA]
    if unknown:
        return jsonify({"error": "unknown locations", "unknown": unknown}), 400

    outbound_map = {CITY_TO_IATA[c]: int(n) for c, n in attendees.items()}

    try:
        window_start = parse_iso_z(window.get("start"))
        window_end = parse_iso_z(window.get("end"))
    except Exception as e:
        return jsonify({"error": "invalid availability_window", "detail": str(e)}), 400

    # evaluate_naive_atOffice only uses days for duration; keep hours separately
    duration_days = int(duration.get("days", 0))

    # call the existing function to pick flights
    try:
        meeting_stats, meeting_stats_by_office = evaluate_naive_atOffice(
            outbound_map, window_start, window_end, duration_days
        )
    except Exception as e:
        return jsonify({"error": "evaluation_failed", "detail": str(e)}), 500

    if not meeting_stats:
        return jsonify({"error": "no_plan_found"}), 500

    viable_meeting_points = [
        (mp, data)
        for mp, data in meeting_stats.items()
        if data
        and data.get("event_dates", {}).get("start") is not None
    ]

    if not viable_meeting_points:
        return jsonify({"error": "no_plan_found"}), 500

    def _score_key(item):
        mp, data = item
        score = data.get("total_score")
        score = float(score) if score is not None else float("inf")
        try:
            priority = OFFICES.index(mp)
        except ValueError:
            priority = len(OFFICES)
        return (score, priority, mp)

    best_meeting_point, best_stats = min(viable_meeting_points, key=_score_key)

    per_office_stats = meeting_stats_by_office.get(best_meeting_point, {})
    attendee_hours_by_office = per_office_stats.get("attendee_travel_hours", {}) or {}
    attendee_co2_by_office = per_office_stats.get("attendee_co2", {}) or {}

    # Map attendee_travel_hours keys back to city names if possible
    iata_to_city = {v: k for k, v in CITY_TO_IATA.items()}

    attendee_travel_hours_named = {}
    for iata, hrs in attendee_hours_by_office.items():
        name = iata_to_city.get(iata, iata)
        try:
            attendee_travel_hours_named[name] = round(float(hrs), 2)
        except Exception:
            attendee_travel_hours_named[name] = hrs

    attendee_co2_named = {}
    for iata, metrics in attendee_co2_by_office.items():
        name = iata_to_city.get(iata, iata)
        attendee_co2_named[name] = metrics

    event_location_code = best_meeting_point
    event_location = IATA_TO_CITY.get(event_location_code, event_location_code)

    avg = best_stats.get("average_travel_hours", 0.0)
    med = best_stats.get("median_travel_hours", 0.0)
    mx = best_stats.get("max_travel_hours", 0.0)
    mn = best_stats.get("min_travel_hours", 0.0)

    result = {
        "event_location": event_location,
        "meeting_point_iata": event_location_code,
        "event_dates": {
            "start": best_stats.get("event_dates", {}).get("start"),
            "end": best_stats.get("event_dates", {}).get("end"),
        },
        "event_span": {
            "start": best_stats.get("event_span", {}).get("start"),
            "end": best_stats.get("event_span", {}).get("end"),
        },
        "total_co2": best_stats.get("total_co2", 0.0),
        "average_travel_hours": avg,
        "median_travel_hours": med,
        "max_travel_hours": mx,
        "min_travel_hours": mn,
        "fairness": best_stats.get("fairness"),
        "total_score": best_stats.get("total_score"),
        "attendee_travel_hours": attendee_travel_hours_named,
        "attendee_co2": attendee_co2_named,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
