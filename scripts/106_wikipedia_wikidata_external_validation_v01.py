#!/usr/bin/env python3
"""
106_wikipedia_wikidata_external_validation_v01.py

Purpose
-------
Perform true external validation using public Wikipedia and Wikidata signals.

For each RDE candidate region, this script queries external public sources
near the region centroid:

- English Wikipedia nearby pages
- Wikidata nearby geotagged entities

No manual scoring.
No invented values.
No paid APIs.

Input
-----
data/processed/rde_external_validation_candidates_v01.csv

Outputs
-------
data/processed/rde_wikipedia_wikidata_external_validation_v01.csv
data/processed/rde_wikipedia_wikidata_external_validation_summary_v01.csv
data/processed/rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv
data/processed/rde_wikipedia_wikidata_external_validation_qa_v01.txt
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SCRIPT_NAME = "106_wikipedia_wikidata_external_validation_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

INPUT_CANDIDATES = PROCESSED / "rde_external_validation_candidates_v01.csv"

OUTPUT_VALIDATION = PROCESSED / "rde_wikipedia_wikidata_external_validation_v01.csv"
OUTPUT_SUMMARY = PROCESSED / "rde_wikipedia_wikidata_external_validation_summary_v01.csv"
OUTPUT_MECH_SUMMARY = PROCESSED / "rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv"
OUTPUT_QA = PROCESSED / "rde_wikipedia_wikidata_external_validation_qa_v01.txt"


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

SEARCH_RADIUS_METERS = 10000
REQUEST_SLEEP_SECONDS = 1.0
TIMEOUT_SECONDS = 20


def load_candidates() -> pd.DataFrame:
    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_CANDIDATES}")

    log.info("Reading candidates: %s", INPUT_CANDIDATES)
    df = pd.read_csv(INPUT_CANDIDATES, low_memory=False)

    required = [
        "mechanism_region_id",
        "canonical_mechanism",
        "validation_centroid_lat",
        "validation_centroid_lon",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate file missing required columns: {missing}")

    log.info("Candidates: %s", len(df))
    return df


def safe_get_json(url: str, params: dict, headers: dict | None = None) -> dict | None:
    default_headers = {
        "User-Agent": "RDE-UREM-validation/1.0 (ethanw263@gmail.com; research use)",
        "Accept": "application/json",
    }

    if headers:
        default_headers.update(headers)

    try:
        r = requests.get(
            url,
            params=params,
            headers=default_headers,
            timeout=TIMEOUT_SECONDS,
        )

        if r.status_code != 200:
            log.warning("Request failed %s: %s", r.status_code, r.text[:150])
            return None

        return r.json()

    except Exception as exc:
        log.warning("Request exception: %s", exc)
        return None


def wikipedia_geosearch(lat: float, lon: float, radius: int = SEARCH_RADIUS_METERS) -> dict:
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": radius,
        "gslimit": 50,
        "format": "json",
    }

    data = safe_get_json(WIKIPEDIA_API, params)

    if not data or "query" not in data:
        return {
            "wiki_nearby_page_count": np.nan,
            "wiki_nearest_page_title": "",
            "wiki_nearest_page_distance_m": np.nan,
            "wiki_top_titles": "",
        }

    pages = data.get("query", {}).get("geosearch", [])

    if not pages:
        return {
            "wiki_nearby_page_count": 0,
            "wiki_nearest_page_title": "",
            "wiki_nearest_page_distance_m": np.nan,
            "wiki_top_titles": "",
        }

    pages_sorted = sorted(pages, key=lambda x: x.get("dist", 999999))

    titles = [p.get("title", "") for p in pages_sorted[:10]]

    return {
        "wiki_nearby_page_count": len(pages),
        "wiki_nearest_page_title": pages_sorted[0].get("title", ""),
        "wiki_nearest_page_distance_m": pages_sorted[0].get("dist", np.nan),
        "wiki_top_titles": " | ".join(titles),
    }


def wikidata_nearby(lat: float, lon: float, radius_km: float = 10.0) -> dict:
    query = f"""
    SELECT ?item ?itemLabel ?dist WHERE {{
      SERVICE wikibase:around {{
        ?item wdt:P625 ?location .
        bd:serviceParam wikibase:center "Point({lon} {lat})"^^geo:wktLiteral .
        bd:serviceParam wikibase:radius "{radius_km}" .
        bd:serviceParam wikibase:distance ?dist .
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY ?dist
    LIMIT 50
    """

    params = {
        "query": query,
        "format": "json",
    }

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "RDE-UREM-validation/1.0 (research script)",
    }

    data = safe_get_json(WIKIDATA_SPARQL, params, headers=headers)

    if not data or "results" not in data:
        return {
            "wikidata_nearby_entity_count": np.nan,
            "wikidata_nearest_entity_label": "",
            "wikidata_nearest_entity_distance_km": np.nan,
            "wikidata_top_labels": "",
        }

    bindings = data.get("results", {}).get("bindings", [])

    if not bindings:
        return {
            "wikidata_nearby_entity_count": 0,
            "wikidata_nearest_entity_label": "",
            "wikidata_nearest_entity_distance_km": np.nan,
            "wikidata_top_labels": "",
        }

    labels = []
    distances = []

    for b in bindings:
        label = b.get("itemLabel", {}).get("value", "")
        dist = b.get("dist", {}).get("value", np.nan)

        labels.append(label)
        try:
            distances.append(float(dist))
        except Exception:
            distances.append(np.nan)

    return {
        "wikidata_nearby_entity_count": len(bindings),
        "wikidata_nearest_entity_label": labels[0] if labels else "",
        "wikidata_nearest_entity_distance_km": distances[0] if distances else np.nan,
        "wikidata_top_labels": " | ".join(labels[:10]),
    }


def normalize_inverse_count(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")

    mn = x.min(skipna=True)
    mx = x.max(skipna=True)

    if pd.isna(mn) or pd.isna(mx) or abs(mx - mn) < 1e-12:
        return pd.Series(0.5, index=x.index)

    return 1 - ((x - mn) / (mx - mn))


def classify_support(score: float) -> str:
    if pd.isna(score):
        return "Insufficient External Data"
    if score >= 0.70:
        return "Strong External Under-Recognition Support"
    if score >= 0.55:
        return "Moderate External Under-Recognition Support"
    if score >= 0.40:
        return "Weak / Mixed External Support"
    return "Low External Under-Recognition Support"


def run_external_queries(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for i, row in df.iterrows():
        rid = row["mechanism_region_id"]
        lat = float(row["validation_centroid_lat"])
        lon = float(row["validation_centroid_lon"])

        log.info("Querying %s (%s/%s)", rid, i + 1, len(df))

        wiki = wikipedia_geosearch(lat, lon)
        time.sleep(REQUEST_SLEEP_SECONDS)

        wd = wikidata_nearby(lat, lon)
        time.sleep(REQUEST_SLEEP_SECONDS)

        out = row.to_dict()
        out.update(wiki)
        out.update(wd)

        rows.append(out)

    return pd.DataFrame(rows)


def score_external_under_recognition(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["wiki_under_recognition_norm"] = normalize_inverse_count(
        df["wiki_nearby_page_count"]
    )

    df["wikidata_under_recognition_norm"] = normalize_inverse_count(
        df["wikidata_nearby_entity_count"]
    )

    if "mean_P_orthogonal_v01" in df.columns:
        p = pd.to_numeric(df["mean_P_orthogonal_v01"], errors="coerce")
    else:
        p = pd.Series(np.nan, index=df.index)

    if "mean_R_net_under_recognition_v01" in df.columns:
        r = pd.to_numeric(df["mean_R_net_under_recognition_v01"], errors="coerce")
    else:
        r = pd.Series(np.nan, index=df.index)

    df["external_validation_physical_component"] = p
    df["external_validation_deficit_component"] = r

    df["wiki_wikidata_external_under_recognition_score"] = df[
        [
            "wiki_under_recognition_norm",
            "wikidata_under_recognition_norm",
            "external_validation_physical_component",
            "external_validation_deficit_component",
        ]
    ].mean(axis=1)

    df["wiki_wikidata_external_validation_class"] = df[
        "wiki_wikidata_external_under_recognition_score"
    ].map(classify_support)

    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "metric": "regions_evaluated",
            "value": len(df),
        },
        {
            "metric": "mean_wikipedia_nearby_page_count",
            "value": float(pd.to_numeric(df["wiki_nearby_page_count"], errors="coerce").mean()),
        },
        {
            "metric": "mean_wikidata_nearby_entity_count",
            "value": float(pd.to_numeric(df["wikidata_nearby_entity_count"], errors="coerce").mean()),
        },
        {
            "metric": "mean_external_under_recognition_score",
            "value": float(df["wiki_wikidata_external_under_recognition_score"].mean()),
        },
        {
            "metric": "strong_or_moderate_support_count",
            "value": int(df["wiki_wikidata_external_validation_class"].isin([
                "Strong External Under-Recognition Support",
                "Moderate External Under-Recognition Support",
            ]).sum()),
        },
    ]

    return pd.DataFrame(rows)


def build_mechanism_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("canonical_mechanism")
        .agg(
            region_count=("mechanism_region_id", "count"),
            mean_wikipedia_pages=("wiki_nearby_page_count", "mean"),
            mean_wikidata_entities=("wikidata_nearby_entity_count", "mean"),
            mean_external_under_recognition_score=(
                "wiki_wikidata_external_under_recognition_score",
                "mean",
            ),
            strong_or_moderate_support_count=(
                "wiki_wikidata_external_validation_class",
                lambda s: int(s.isin([
                    "Strong External Under-Recognition Support",
                    "Moderate External Under-Recognition Support",
                ]).sum()),
            ),
        )
        .reset_index()
        .sort_values("mean_external_under_recognition_score", ascending=False)
    )


def trim_output(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "mechanism_region_id",
        "canonical_mechanism",
        "external_validation_priority_score",
        "external_validation_priority_tier",
        "validation_centroid_lat",
        "validation_centroid_lon",
        "mean_P_orthogonal_v01",
        "mean_R_net_under_recognition_v01",
        "wiki_nearby_page_count",
        "wiki_nearest_page_title",
        "wiki_nearest_page_distance_m",
        "wiki_top_titles",
        "wikidata_nearby_entity_count",
        "wikidata_nearest_entity_label",
        "wikidata_nearest_entity_distance_km",
        "wikidata_top_labels",
        "wiki_under_recognition_norm",
        "wikidata_under_recognition_norm",
        "wiki_wikidata_external_under_recognition_score",
        "wiki_wikidata_external_validation_class",
    ]

    return df[[c for c in keep if c in df.columns]].copy()


def write_outputs(df: pd.DataFrame, summary: pd.DataFrame, mech_summary: pd.DataFrame) -> None:
    trimmed = trim_output(df)

    log.info("Writing validation: %s", OUTPUT_VALIDATION)
    trimmed.to_csv(OUTPUT_VALIDATION, index=False)

    log.info("Writing summary: %s", OUTPUT_SUMMARY)
    summary.to_csv(OUTPUT_SUMMARY, index=False)

    log.info("Writing mechanism summary: %s", OUTPUT_MECH_SUMMARY)
    mech_summary.to_csv(OUTPUT_MECH_SUMMARY, index=False)

    qa = []
    qa.append("RDE Wikipedia/Wikidata External Validation v01 QA")
    qa.append("=" * 55)
    qa.append("")
    qa.append("Summary:")
    qa.append(summary.to_string(index=False))
    qa.append("")
    qa.append("Mechanism summary:")
    qa.append(mech_summary.to_string(index=False))
    qa.append("")
    qa.append("Validation class counts:")
    qa.append(trimmed["wiki_wikidata_external_validation_class"].value_counts().to_string())
    qa.append("")
    qa.append("Top 25 external under-recognition candidates:")
    qa.append(
        trimmed.sort_values("wiki_wikidata_external_under_recognition_score", ascending=False)
        .head(25)
        .to_string(index=False)
    )

    log.info("Writing QA: %s", OUTPUT_QA)
    OUTPUT_QA.write_text("\n".join(qa), encoding="utf-8")


def main() -> None:
    log.info("Starting Script 106: Wikipedia/Wikidata external validation")

    candidates = load_candidates()
    queried = run_external_queries(candidates)
    scored = score_external_under_recognition(queried)

    summary = build_summary(scored)
    mech_summary = build_mechanism_summary(scored)

    write_outputs(scored, summary, mech_summary)

    log.info("Done")

    print("\nWikipedia/Wikidata External Validation Summary:")
    print(summary.to_string(index=False))

    print("\nMechanism Summary:")
    print(mech_summary.to_string(index=False))

    print("\nValidation Class Counts:")
    print(scored["wiki_wikidata_external_validation_class"].value_counts().to_string())

    print("\nCreated:")
    print(f"  {OUTPUT_VALIDATION}")
    print(f"  {OUTPUT_SUMMARY}")
    print(f"  {OUTPUT_MECH_SUMMARY}")
    print(f"  {OUTPUT_QA}")


if __name__ == "__main__":
    main()