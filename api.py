from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
import math
import numpy as np

from main import get_flights_score, evaluate_naive_atOffice, OFFICES

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


def to_iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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
    duration_hours = int(duration.get("hours", 0))

    # call the existing function to pick flights
    try:
        final_flights = evaluate_naive_atOffice(outbound_map, window_start, window_end, duration_days)
    except Exception as e:
        return jsonify({"error": "evaluation_failed", "detail": str(e)}), 500

    if not final_flights:
        return jsonify({"error": "no_plan_found"}), 500

    # Extract event location: arrival city if present, otherwise arrival airport (IATA)
    sample = None
    for v in final_flights.values():
        sample = v
        break

    event_location = None
    if sample is not None:
        # prefer ARRCITY, then INTAPT, then infer from OFFICES
        if "ARRCITY" in sample and sample["ARRCITY"]:
            event_location = sample["ARRCITY"][0]
        elif "INTAPT" in sample and sample["INTAPT"]:
            event_location = sample["INTAPT"][0]
    if event_location is None:
        # fallback: pick first office code
        event_location = OFFICES[0]

    event_location = IATA_TO_CITY.get(event_location, event_location)

    # compute attendee travel hours and co2
    per_attendee_hours = []
    attendee_travel_hours = {}
    total_co2 = 0.0

    for origin_iata, info in final_flights.items():
        count = int(outbound_map.get(origin_iata, 1))
        # data from main.get_flights_score is returned as lists for each column
        elptim_val = None
        co2_per_capita = None
        if isinstance(info.get("ELPTIM"), list) and info.get("ELPTIM"):
            elptim_val = info.get("ELPTIM")[0]
        else:
            elptim_val = info.get("ELPTIM")

        if isinstance(info.get("ESTIMATED_CO2_PER_CAPITA"), list) and info.get("ESTIMATED_CO2_PER_CAPITA"):
            co2_per_capita = info.get("ESTIMATED_CO2_PER_CAPITA")[0]
        else:
            co2_per_capita = info.get("ESTIMATED_CO2_PER_CAPITA")

        # assume ELPTIM is minutes (common in schedule datasets); convert to hours
        try:
            elptim_val = float(elptim_val)
            hours = elptim_val / 60.0
        except Exception:
            # fallback: if it's already hours
            try:
                hours = float(elptim_val)
            except Exception:
                hours = 0.0

        attendee_travel_hours[origin_iata] = round(hours, 2)
        # add to per-individual list
        per_attendee_hours.extend([hours] * count)

        # CO2 per capita is in tonnes; sum across attendees
        try:
            co2_per_capita = float(co2_per_capita)
        except Exception:
            co2_per_capita = 0.0

        total_co2 += co2_per_capita * count

    if per_attendee_hours:
        avg = float(np.mean(per_attendee_hours))
        med = float(np.median(per_attendee_hours))
        mx = float(np.max(per_attendee_hours))
        mn = float(np.min(per_attendee_hours))
    else:
        avg = med = mx = mn = 0.0

    # Build event date/time estimates
    # Choose event start as availability_window start + 30 minutes (heuristic)
    event_start = window_start + timedelta(minutes=30)
    event_end = event_start + timedelta(days=duration_days, hours=duration_hours)

    # event span extended by max travel hours on either side
    event_span_start = event_start - timedelta(hours=mx)
    event_span_end = event_end + timedelta(hours=mx)

    # Map attendee_travel_hours keys back to city names if possible
    iata_to_city = {v: k for k, v in CITY_TO_IATA.items()}
    attendee_travel_hours_named = {}
    for iata, hrs in attendee_travel_hours.items():
        name = iata_to_city.get(iata, iata)
        attendee_travel_hours_named[name] = hrs

    result = {
        "event_location": event_location,
        "event_dates": {
            "start": to_iso_z(event_start),
            "end": to_iso_z(event_end),
        },
        "event_span": {
            "start": to_iso_z(event_span_start),
            "end": to_iso_z(event_span_end),
        },
        "total_co2": round(float(total_co2), 3),
        "average_travel_hours": round(avg, 2),
        "median_travel_hours": round(med, 2),
        "max_travel_hours": round(mx, 2),
        "min_travel_hours": round(mn, 2),
        "attendee_travel_hours": attendee_travel_hours_named,
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
