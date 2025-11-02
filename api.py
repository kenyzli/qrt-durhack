from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
import json
import math
import os
import numpy as np
import google.generativeai as genai

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

GEMINI_MODEL_NAME = "gemma-3n-e2b-it"
DEFAULT_GEMINI_PROMPT = (
    "You are an expert travel coordinator. Using ONLY the data in the JSON below, "
    "write a concise executive summary that explains the trade-offs and gives a brief snapshot of the destination.\n\n"
    "OUTPUT REQUIREMENTS\n"
    "- Max 180 words.\n"
    "- Plain English, scannable bullets.\n"
    "- Do NOT invent numbers or facts not present in the JSON. Use values exactly as given.\n"
    "- You may rank or compare cities using the provided travel hours, but do not compute percentages.\n"
    "- If something isn't in the JSON, omit it (don't guess).\n\n"
    "COVER EXACTLY THESE SECTIONS\n\n"
    "1) When & Where\n"
    "- Event location and the event date range (use the ISO strings as-is).\n\n"
    "2) Destination Snapshot\n"
    "- 2-3 high-level, generic points about the location (e.g., major hub status, infrastructure, accessibility). "
    "Avoid weather, visas, or speculative info. Keep it qualitative and factual.\n\n"
    "3) Travel Burden & Fairness\n"
    "- State average, median, min, and max travel hours.\n"
    "- Name up to 2 longest-haul origins and up to 2 shortest-haul origins (from attendee_travel_hours), with hours.\n\n"
    "4) Carbon Impact\n"
    "- Report total_co2 from the JSON.\n"
    "- Briefly note how travel time distribution might relate to emissions (qualitative only).\n\n"
    "5) Key Trade-offs (3 bullets)\n"
    "- Balance of fairness (who travels most/least), total time, and carbon.\n"
    "- Mention any notable outliers or risks (e.g., very long itineraries).\n"
)

# _gemini_model = None


# class GeminiSummaryConfigurationError(RuntimeError):
#     """Raised when Gemini is not configured correctly."""


# def _get_gemini_model():
#     global _gemini_model
#     api_key = os.environ.get("GOOGLE_API_KEY")
#     if not api_key:
#         raise GeminiSummaryConfigurationError("GOOGLE_API_KEY is not set")
#     # Configure once and reuse the GenerativeModel instance.
#     genai.configure(api_key=api_key)
#     if _gemini_model is None:
#         _gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
#     return _gemini_model


# def generate_gemini_summary(result_payload, prompt_override=None):
#     prompt = prompt_override or DEFAULT_GEMINI_PROMPT
#     payload_json = json.dumps(result_payload, default=str)
#     prompt_text = f"{prompt}\n\nJSON:\n{payload_json}"

#     model = _get_gemini_model()

#     response = model.generate_content(prompt_text)
#     summary_text = getattr(response, "text", None) or str(response)
#     return summary_text


# @app.route("/gemini_summary", methods=["POST"])
# def gemini_summary():
#     """
#     Expects JSON body:
#       {
#         "result": { ... your /plan result ... },
#         "prompt": "optional custom prompt"
#       }

#     Returns:
#       {
#         "summary": "Gemini-generated text summary",
#       }
#     """
#     data = request.get_json()
#     if not data or "result" not in data:
#         return jsonify({"error": "missing 'result' field"}), 400

#     result = data["result"]
#     prompt_override = data.get("prompt")

#     try:
#         summary_text = generate_gemini_summary(result, prompt_override=prompt_override)
#     except GeminiSummaryConfigurationError as exc:
#         return jsonify({
#             "error": "missing_api_key",
#             "detail": str(exc),
#         }), 500
#     except Exception as exc:  # pragma: no cover - passes through backend error detail
#         return jsonify({
#             "error": "gemini_request_failed",
#             "detail": str(exc),
#         }), 500

#     return jsonify({
#         "summary": summary_text
#     })


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

    sorted_meeting_points = sorted(viable_meeting_points, key=_score_key)

    iata_to_city = {v: k for k, v in CITY_TO_IATA.items()}

    # gemini_summary_enabled = True
    # gemini_setup_error = None
    # try:
    #     _get_gemini_model()
    # except Exception as exc:
    #     gemini_summary_enabled = False
    #     gemini_setup_error = str(exc)

    meeting_options = []
    for meeting_point, stats in sorted_meeting_points:
        per_office_stats = meeting_stats_by_office.get(meeting_point, {})
        attendee_hours_by_office = per_office_stats.get("attendee_travel_hours", {}) or {}
        attendee_co2_by_office = per_office_stats.get("attendee_co2", {}) or {}
        attendee_routes_by_office = per_office_stats.get("attendee_routes", {}) or {}

        attendee_travel_hours_named = {}
        for iata, hrs in attendee_hours_by_office.items():
            name = iata_to_city.get(iata, iata)
            try:
                attendee_travel_hours_named[name] = round(float(hrs), 2)
            except Exception:
                attendee_travel_hours_named[name] = hrs
        
        attendee_routes_named = {}
        for iata, route_nodes in attendee_routes_by_office.items():
            name = iata_to_city.get(iata, iata)
            attendee_routes_named[name] = route_nodes

        attendee_co2_named = {}
        for iata, metrics in attendee_co2_by_office.items():
            name = iata_to_city.get(iata, iata)
            attendee_co2_named[name] = metrics

        event_location = IATA_TO_CITY.get(meeting_point, meeting_point)

        option = {
            "event_location": event_location,
            "meeting_point_iata": meeting_point,
            "event_dates": {
                "start": stats.get("event_dates", {}).get("start"),
                "end": stats.get("event_dates", {}).get("end"),
            },
            "event_span": {
                "start": stats.get("event_span", {}).get("start"),
                "end": stats.get("event_span", {}).get("end"),
            },
            "total_co2": stats.get("total_co2", 0.0),
            "average_travel_hours": stats.get("average_travel_hours", 0.0),
            "median_travel_hours": stats.get("median_travel_hours", 0.0),
            "max_travel_hours": stats.get("max_travel_hours", 0.0),
            "min_travel_hours": stats.get("min_travel_hours", 0.0),
            "fairness": stats.get("fairness"),
            "total_score": stats.get("total_score"),
            "attendee_travel_hours": attendee_travel_hours_named,
            "attendee_co2": attendee_co2_named,
            "attendee_routes": attendee_routes_named,
        }

        # summary_text = None
        # summary_error = None
        # if gemini_summary_enabled:
        #     try:
        #         summary_text = generate_gemini_summary(option)
        #     except Exception as exc:
        #         summary_error = str(exc)
        # else:
        #     summary_error = gemini_setup_error

        # option["gemini_summary"] = summary_text
        # if summary_error:
            # option["gemini_summary_error"] = summary_error

        meeting_options.append(option)

    return jsonify(meeting_options)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
