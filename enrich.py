#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

import pandas as pd


def load_input_csv(path: Path) -> pd.DataFrame:
    # Handle weird quotes/BOMs like “Hartsfield�Jackson…” by using utf-8-sig
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="cp1252")
    # Normalize expected columns
    # Your sample has a column literally named: code\n(iata/icao)
    # We'll find the first column containing both 'iata' and 'icao'
    code_col = None
    for c in df.columns:
        c_norm = c.lower().replace(" ", "")
        if "iata" in c_norm and "icao" in c_norm:
            code_col = c
            break
    if not code_col:
        raise SystemExit("Could not find a column containing both 'iata' and 'icao' in the header.")

    # Extract IATA and ICAO from values like "ATL/KATL"
    def parse_codes(val: str):
        if pd.isna(val):
            return pd.Series({"iata_code_in": None, "icao_code_in": None})
        # Look for patterns like "ATL/KATL", possibly with spaces
        m = re.match(r"\s*([A-Z0-9]{2,4})\s*/\s*([A-Z0-9]{3,4})\s*", str(val))
        if m:
            iata, icao = m.group(1), m.group(2)
            # Heuristic: IATA is usually 3 letters; ICAO 4.
            if len(iata) == 4 and len(icao) == 3:
                # Swap if reversed
                iata, icao = icao, iata
            return pd.Series({"iata_code_in": iata, "icao_code_in": icao})
        # Fallback: if there's only one token, try to infer by length
        tok = re.sub(r"\s+", "", str(val))
        if len(tok) == 3:
            return pd.Series({"iata_code_in": tok, "icao_code_in": None})
        if len(tok) == 4:
            return pd.Series({"iata_code_in": None, "icao_code_in": tok})
        return pd.Series({"iata_code_in": None, "icao_code_in": None})

    parsed = df[code_col].apply(parse_codes)
    df = pd.concat([df, parsed], axis=1)
    return df


def load_ourairports(path: Path) -> pd.DataFrame:
    # airports.csv columns include: ident (ICAO), iata_code, name, latitude_deg, longitude_deg, type, iso_country, …
    gaz = pd.read_csv(path, encoding="utf-8-sig")
    # Normalize columns we need
    needed = ["ident", "iata_code", "name", "latitude_deg", "longitude_deg"]
    missing = [c for c in needed if c not in gaz.columns]
    if missing:
        raise SystemExit(f"OurAirports file is missing columns: {missing}")
    gaz.rename(
        columns={"ident": "icao_code", "iata_code": "iata_code_our", "name": "airport_name"},
        inplace=True,
    )
    # For speed, keep only a few columns
    gaz = gaz[["icao_code", "iata_code_our", "airport_name", "latitude_deg", "longitude_deg"]].copy()
    # Make sure codes are strings uppercased
    gaz["icao_code"] = gaz["icao_code"].astype(str).str.strip().str.upper()
    gaz["iata_code_our"] = gaz["iata_code_our"].astype(str).str.strip().str.upper().replace({"NAN": None})
    return gaz


def enrich(df_in: pd.DataFrame, gaz: pd.DataFrame) -> pd.DataFrame:
    # Try IATA match first
    # normalize to uppercase first
    df_in["iata_code_in"] = df_in["iata_code_in"].astype(str).str.upper()
    df_in["icao_code_in"] = df_in["icao_code_in"].astype(str).str.upper()
    gaz["iata_code_our"] = gaz["iata_code_our"].astype(str).str.upper()
    gaz["icao_code"] = gaz["icao_code"].astype(str).str.upper()

    # then merge by column name
    left1 = df_in.merge(
        gaz,
        left_on="iata_code_in",
        right_on="iata_code_our",
        how="left",
        suffixes=("", "_by_iata"),
    )
    need_icao = left1["latitude_deg"].isna() | left1["longitude_deg"].isna()
    if need_icao.any():
        left_missing = left1[need_icao].drop(columns=["icao_code", "iata_code_our", "airport_name", "latitude_deg", "longitude_deg"])
        icao_merge = left_missing.merge(
            gaz,
            left_on=left_missing["icao_code_in"].str.upper(),
            right_on=gaz["icao_code"],
            how="left",
        )
        # Fill lat/lon where missing
        for col in ["latitude_deg", "longitude_deg", "airport_name"]:
            left1.loc[need_icao, col] = left1.loc[need_icao, col].fillna(icao_merge[col])

    # Clean up and rename outputs
    out = left1.copy()
    out.rename(
        columns={
            "latitude_deg": "latitude",
            "longitude_deg": "longitude",
            "airport_name": "matched_airport_name",
        },
        inplace=True,
    )

    # Report any not found
    not_found = out["latitude"].isna() | out["longitude"].isna()
    if not_found.any():
        sys.stderr.write("WARN: Could not locate coordinates for these rows (by code):\n")
        cols_to_show = ["airport", "location", "country", "iata_code_in", "icao_code_in"]
        for _, row in out[not_found][[c for c in cols_to_show if c in out.columns]].iterrows():
            sys.stderr.write(f" - {row.to_dict()}\n")

    return out


def main():
    p = argparse.ArgumentParser(
        description="Enrich an airports CSV with latitude/longitude using the OurAirports dataset."
    )
    p.add_argument(
        "-i", "--input",
        required=True,
        help="Path to your input CSV (the one with 'code (iata/icao)' column).",
    )
    p.add_argument(
        "-g", "--gazetteer",
        required=True,
        help="Path to OurAirports airports.csv (download once; no API key needed).",
    )
    p.add_argument(
        "-o", "--output",
        required=True,
        help="Where to write the enriched CSV.",
    )
    p.add_argument(
        "--keep-cols",
        default=None,
        help="Comma-separated list of extra columns to keep from the OurAirports file (e.g., 'type,iso_country').",
    )

    args = p.parse_args()
    df_in = load_input_csv(Path(args.input))
    gaz = load_ourairports(Path(args.gazetteer))

    if args.keep_cols:
        extras = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
        # Merge extras if they exist
        extras_ok = [c for c in extras if c in gaz.columns]
        gaz = gaz[["icao_code", "iata_code_our", "airport_name", "latitude_deg", "longitude_deg"] + extras_ok]

    enriched = enrich(df_in, gaz)
    enriched.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
