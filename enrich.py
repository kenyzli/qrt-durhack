from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [
    BASE_DIR / "airport_data",
    BASE_DIR / "data",
    BASE_DIR / "1",
    BASE_DIR,
]


def _sanitize_header(raw: str) -> str:
    cleaned = raw.replace("\r", " ").replace("\n", " ").strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    cleaned = "_".join(segment for segment in cleaned.split() if segment)
    return cleaned or "column"


def _unique_headers(headers: Iterable[str]) -> List[str]:
    seen: Dict[str, int] = {}
    result: List[str] = []
    for header in headers:
        base = _sanitize_header(header)
        count = seen.get(base, 0)
        candidate = f"{base}_{count}" if count else base
        seen[base] = count + 1
        result.append(candidate)
    return result


def _find_file(filename: str) -> Path:
    for directory in DATA_DIR_CANDIDATES:
        candidate = directory / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate '{filename}' in {DATA_DIR_CANDIDATES}")


def load_busiest_rankings(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="cp1252") as handle:
        reader = csv.reader(handle)
        try:
            raw_header = next(reader)
        except StopIteration:
            return []
        headers = _unique_headers(raw_header)
        rows: List[Dict[str, str]] = []
        for values in reader:
            if not any(values):
                continue
            rows.append(dict(zip(headers, values)))
    return rows


def load_airport_coordinates(path: Path) -> Dict[str, Tuple[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        coords: Dict[str, Tuple[str, str]] = {}
        for row in reader:
            iata_code = (row.get("iata_code") or "").strip().upper()
            if not iata_code or iata_code in coords:
                continue
            latitude = (row.get("latitude_deg") or "").strip()
            longitude = (row.get("longitude_deg") or "").strip()
            if not latitude or not longitude:
                continue
            coords[iata_code] = (latitude, longitude)
    return coords


def extract_iata(code_field: str) -> str:
    value = (code_field or "").strip()
    if not value:
        return ""
    return value.split("/")[0].strip().upper()


def build_rows(busiest_path: Path, airports_path: Path) -> List[Tuple[str, str, str]]:
    rankings = load_busiest_rankings(busiest_path)
    if not rankings:
        return []
    coords_lookup = load_airport_coordinates(airports_path)
    rows: List[Tuple[str, str, str]] = []
    for entry in rankings:
        airport_name = (entry.get("airport") or "").strip()
        if not airport_name:
            continue
        code_field = entry.get("code_iata_icao") or ""
        iata_code = extract_iata(code_field)
        if not iata_code:
            continue
        coords = coords_lookup.get(iata_code)
        if not coords:
            continue
        lat, lon = coords
        rows.append((airport_name, f"{lat},{lon}", iata_code))
    return rows


def write_output(rows: List[Tuple[str, str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["airport", "coordinates", "code"])
        writer.writerows(rows)


def resolve_paths() -> Tuple[Path, Path, Path]:
    busiest_path = _find_file("busiestAirports.csv")
    airports_path = _find_file("airports.csv")
    output_path = busiest_path.with_name("busiest_airports_coordinates.csv")
    return busiest_path, airports_path, output_path


def main() -> None:
    busiest_path, airports_path, output_path = resolve_paths()
    rows = build_rows(busiest_path, airports_path)
    if not rows:
        raise SystemExit("No matching airport rows were produced; check input datasets.")
    write_output(rows, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
