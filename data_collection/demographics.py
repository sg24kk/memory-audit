"""
S1.4 -- gender and ethnicity attributes for every scholar.

Both are waterfalls: each scholar's value comes from exactly ONE source,
chosen by a fixed order of preference. No cross-tool voting, no disagreement
column to interpret downstream.

GENDER
    1. Wikidata P21, matched by ORCID     (recorded fact, not inference)
    2. gender-guesser                     (dictionary lookup)
    3. nomquamgender                      (name-to-gender probabilities)
    4. unknown

    Wikidata covers ~27% of the sample: 68% have an ORCID, and ~40% of those
    resolve to a Wikidata entry with gender set. Coverage is strongly
    fame-correlated (10% in the lowest h-index band, 82% in the highest), so
    it is used to IMPROVE labels where available, never to decide who stays
    in the sample -- otherwise it would bias intake toward famous scholars.

ETHNICITY
    ethnicseer first (12 origin categories), ethnicolr2 as the fallback
    (5 US-census categories). Measured on a 10-name probe: ethnicseer 8/10,
    ethnicolr2 6/10, and they fail on OPPOSITE populations -- ethnicolr2 is
    the only one that gets African names right (Kwame Nkrumah, 0.969), while
    ethnicseer is the only one that gets French, Indian and Middle-Eastern
    right. Hence the waterfall rather than either alone.

    Neither tool's confidence identifies its own errors: ethnicolr2 is 0.906
    confident that a Tunisian name is nh_white, ethnicseer 0.881 confident
    it is French. Confidence is stored for sensitivity analysis but is NOT a
    correctness filter.
"""

from __future__ import annotations

import glob
import json
import re
import time
import warnings
from pathlib import Path

import pandas as pd
import requests

from memory_audit.data_collection.names import resolve_full_name

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "memory-audit/0.1 (research; khaled.gardabbou@etudiant-enit.utm.tn)"

# Accept ethnicseer above this; below it, consult ethnicolr2.
ETHNICSEER_ACCEPT = 0.55
# Below this, an ethnicolr2 answer is treated as too weak to override a
# disagreeing ethnicseer label.
ETHNICOLR2_WEAK = 0.40

# Which ethnicseer labels correspond to which ethnicolr2 category, used to
# decide whether the two tools "agree". `mea` is deliberately absent: US
# census categories have no Middle-Eastern bucket, so it can never match.
SEER_TO_CENSUS = {
    "eng": "nh_white", "frn": "nh_white", "ger": "nh_white",
    "ita": "nh_white", "rus": "nh_white",
    "spa": "hispanic",
    "chi": "asian", "ind": "asian", "jap": "asian",
    "kor": "asian", "vie": "asian",
    "mea": "mena",
}
WHITE_FAMILY = {"eng", "frn", "ger", "ita", "rus"}


# ----------------------------------------------------------------- gender --

def _sparql(query: str, retries: int = 4) -> dict | None:
    """Run one SPARQL query, retrying transient failures.

    Wikidata returns 502s under load. Without retries a failed batch
    silently drops ~150 scholars to the name-based layers, and a rerun
    would fail on DIFFERENT batches -- so labels would not be reproducible.
    """
    for attempt in range(retries):
        try:
            r = requests.get(
                WIKIDATA_SPARQL,
                params={"query": query, "format": "json"},
                headers={"User-Agent": USER_AGENT},
                timeout=90,
            )
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 ** attempt)
    return None


WIKIDATA_CACHE = RESULTS / "wikidata_gender.json"


def wikidata_gender(orcids: list[str], batch: int = 150) -> dict[str, str]:
    """ORCID -> gender, for the subset Wikidata knows about.

    Batched into one SPARQL VALUES query per chunk; a scholar with no
    Wikidata entry simply does not come back in the results.

    Only male/female are recorded. Wikidata carries other values, but the
    gender column is binary by design, so anything else is left unmapped and
    falls through to the name-based layers rather than creating a third class.

    Results are cached to disk. The lookup keys on ORCID, so it is unaffected
    by anything we do to names -- without the cache every rerun spends six
    minutes re-fetching identical answers, and each attempt risks a fresh set
    of transient 502s making the labels differ between runs.
    """
    cached: dict[str, str] = {}
    if WIKIDATA_CACHE.exists():
        cached = json.loads(WIKIDATA_CACHE.read_text(encoding="utf-8"))
        missing = [o for o in orcids if o not in cached]
        if not missing:
            print(f"  Wikidata: {len(cached):,} answers from cache, no queries needed")
            return {o: cached[o] for o in orcids if o in cached}
        print(f"  Wikidata: {len(cached):,} cached, querying {len(missing):,} new")
        orcids = missing

    found: dict[str, str] = {}
    for i in range(0, len(orcids), batch):
        values = " ".join(f'"{o}"' for o in orcids[i:i + batch])
        payload = _sparql(f"""
        SELECT ?orcid ?genderLabel WHERE {{
          VALUES ?orcid {{ {values} }}
          ?person wdt:P496 ?orcid.
          ?person wdt:P21 ?gender.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        """)
        if payload is None:
            print(f"  wikidata batch {i//batch}: failed after retries, skipped")
            continue
        for row in payload["results"]["bindings"]:
            if row["genderLabel"]["value"].lower() in {"male", "female"}:
                found[row["orcid"]["value"]] = row["genderLabel"]["value"].lower()
        time.sleep(1.0)

    cached.update(found)
    WIKIDATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    WIKIDATA_CACHE.write_text(json.dumps(cached), encoding="utf-8")
    return cached


def add_resolved_name(df: pd.DataFrame) -> pd.DataFrame:
    """Recover full first names from OpenAlex's name variants.

    `display_name` is initials-only for 15.7% of the sample and 66.4% of
    physics, which is what drives the unknown-gender rate. See names.py for
    the matching rules and the merged-record risk they guard against.

    `display_name` is left untouched -- the expansion lands in
    `resolved_name`, with `name_status` recording how it got there, so any
    label can be traced back to the name it was actually inferred from.
    """
    def alternatives(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                return []
        return list(value) if value is not None else []

    resolved = [
        resolve_full_name(name, alternatives(alts))
        for name, alts in zip(df["display_name"], df["display_name_alternatives"])
    ]
    df = df.copy()
    df["resolved_name"] = [r[0] for r in resolved]
    df["name_status"] = [r[1] for r in resolved]
    n_expanded = (df["name_status"] == "expanded").sum()
    print(f"  recovered full first names for {n_expanded:,} scholars")
    return df


def _first_name(full: str) -> str:
    parts = str(full).split()
    return parts[0] if parts else ""


# A standalone initial: one letter, optionally followed by a period.
_INITIAL = re.compile(r"^[A-Za-z]\.?$")


def split_name(full: str) -> tuple[str, str]:
    """Split a display name into (first, last), dropping middle initials.

    Naive `split(n=1)` hands everything after the first name to the
    classifier as a surname, so "Paul H. Rabinowitz" is scored as
    "H. Rabinowitz" -- which ethnicolr2 reads as `other` (0.479) where the
    bare surname gives `nh_white` (0.913). 28.7% of the sample carries a
    middle initial, so this is not a rare edge case.

    Taking only the LAST token would fix that but break compound surnames
    ("Jan van der Berg" -> "Berg"), so instead the initials are removed and
    everything else after the first name is kept.
    """
    tokens = str(full).split()
    if not tokens:
        return "", "Unknown"
    first = tokens[0]
    rest = [t for t in tokens[1:] if not _INITIAL.match(t)]
    return first, " ".join(rest) if rest else "Unknown"


def assign_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Add gender + the source that decided it."""
    import gender_guesser.detector as gg
    import nomquamgender as nqg

    df = df.copy()
    df["_first"] = df["resolved_name"].map(_first_name)

    # --- layer 1: Wikidata, via ORCID
    orcids = (
        df.loc[df["orcid"].notna(), "orcid"]
        .str.rsplit("/", n=1).str[-1].unique().tolist()
    )
    print(f"  querying Wikidata for {len(orcids):,} ORCIDs...")
    wd = wikidata_gender(orcids)
    print(f"  Wikidata returned gender for {len(wd):,}")

    df["_orcid_id"] = df["orcid"].str.rsplit("/", n=1).str[-1]
    gender = df["_orcid_id"].map(wd)
    source = pd.Series(pd.NA, index=df.index, dtype="object")
    source[gender.notna()] = "wikidata"

    # --- layer 2: gender-guesser
    detector = gg.Detector(case_sensitive=False)
    gg_map = {"male": "male", "mostly_male": "male",
              "female": "female", "mostly_female": "female"}
    need = gender.isna()
    gg_raw = df.loc[need, "_first"].map(lambda n: detector.get_gender(n) if n else "unknown")
    gg_lab = gg_raw.map(gg_map)
    gender[need] = gg_lab
    source[need & gg_lab.notna()] = "gender_guesser"

    # --- layer 3: nomquamgender
    need = gender.isna()
    if need.any():
        model = nqg.NBGC()
        names = df.loc[need, "_first"].tolist()
        ann = model.annotate(names, as_df=True)
        # p(gf) is P(female); no estimate when the name is unknown to it.
        p_female = pd.Series(ann["p(gf)"].values, index=df.index[need])
        nqg_lab = p_female.map(
            lambda p: pd.NA if pd.isna(p) else ("female" if p >= 0.5 else "male")
        )
        gender[need] = nqg_lab
        source[need & nqg_lab.notna()] = "nomquamgender"

    df["gender"] = gender.fillna("unknown")
    df["gender_source"] = source.fillna("none")
    return df.drop(columns=["_first", "_orcid_id"])


# -------------------------------------------------------------- ethnicity --

def _to_census(seer_lab: str) -> str:
    """ethnicseer label -> census category, refusing to invent a null.

    Every one of ethnicseer's twelve labels is mapped. A KeyError here means
    the tool's vocabulary changed, which should stop the run rather than
    quietly write None into the ethnicity column.
    """
    try:
        return SEER_TO_CENSUS[seer_lab]
    except KeyError:
        raise KeyError(
            f"ethnicseer returned unmapped label {seer_lab!r}; "
            f"add it to SEER_TO_CENSUS"
        ) from None


def _resolve_ethnicity(seer_lab, seer_conf, olr_lab, olr_conf):
    """The waterfall, exactly as specified.

    Returns (label, source). `label` uses census vocabulary so the column is
    single-valued; the fine-grained ethnicseer label is kept separately.
    """
    # ethnicseer is confident enough on its own.
    # All twelve of its labels are in SEER_TO_CENSUS, so a miss would mean
    # the tool changed under us -- fail loudly rather than write a null.
    if seer_conf >= ETHNICSEER_ACCEPT:
        return _to_census(seer_lab), "ethnicseer"

    # ethnicseer was weak -- consult ethnicolr2
    # (a) only ethnicolr2 can express Black at all
    if olr_lab == "nh_black":
        return "nh_black", "ethnicolr2"

    # (b) both point at the white family: keep ethnicseer's finer read
    if olr_lab == "nh_white" and seer_lab in WHITE_FAMILY:
        return "nh_white", "ethnicseer"

    # (c) ethnicolr2 is itself weak AND disagrees: prefer ethnicseer
    if olr_conf < ETHNICOLR2_WEAK and SEER_TO_CENSUS.get(seer_lab) != olr_lab:
        return _to_census(seer_lab), "ethnicseer"

    # ethnicolr2 has a sixth "other" bucket with no counterpart in
    # ethnicseer's twelve. Falling back to it would leave scholars in a
    # category that means nothing -- observed on names both tools find hard
    # (Uyghur, Persian, accented Spanish). Take ethnicseer's read instead,
    # low confidence and all, so every scholar lands in a real category.
    if olr_lab == "other":
        return _to_census(seer_lab), "ethnicseer"

    # default: we only got here because ethnicseer was weak
    return olr_lab, "ethnicolr2"


def assign_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    """Add ethnicity, its source, both tools' raw labels and confidences."""
    from ethnicolr2 import pred_fl_full_name
    from ethnicseer import EthnicClassifier

    df = df.copy()
    names = df["resolved_name"].astype(str).tolist()

    print(f"  ethnicseer on {len(names):,} names...")
    ec = EthnicClassifier.load_pretrained_model()
    seer_labels, seer_scores = ec.classify_names_with_scores(names)
    df["ethnicseer_label"] = seer_labels
    df["ethnicseer_conf"] = [float(s) for s in seer_scores]

    print(f"  ethnicolr2 on {len(names):,} names...")
    split = [split_name(n) for n in names]
    probe = pd.DataFrame({
        "first": [s[0] for s in split],
        "last": [s[1] for s in split],
    })
    olr = pred_fl_full_name(probe, "last", "first").reset_index(drop=True)
    df["ethnicolr2_label"] = olr["preds"].values
    df["ethnicolr2_conf"] = [float(max(p.values())) for p in olr["probs"]]

    resolved = [
        _resolve_ethnicity(sl, sc, ol, oc)
        for sl, sc, ol, oc in zip(
            df["ethnicseer_label"], df["ethnicseer_conf"],
            df["ethnicolr2_label"], df["ethnicolr2_conf"],
        )
    ]
    df["ethnicity"] = [r[0] for r in resolved]
    df["ethnicity_source"] = [r[1] for r in resolved]
    return df


# ----------------------------------------------------------------- report --

def report(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\n{n:,} scholars\n")

    print("--- gender ---")
    print(df["gender"].value_counts().to_string())
    print("\n  by source:")
    print(df["gender_source"].value_counts().to_string())

    print("\n--- ethnicity ---")
    print(df["ethnicity"].value_counts().to_string())
    print("\n  decided by:")
    print(df["ethnicity_source"].value_counts().to_string())

    print("\n  fine-grained (ethnicseer, for reference):")
    print(df["ethnicseer_label"].value_counts().to_string())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ethnicity-only", action="store_true",
        help="Reuse existing gender columns instead of re-querying Wikidata "
             "(~6 min of SPARQL). Ethnicity runs locally in seconds.",
    )
    args = parser.parse_args()

    out = RESULTS / "sample_demographics.parquet"

    if args.ethnicity_only:
        if not out.exists():
            raise SystemExit(f"{out.name} not found -- run without --ethnicity-only first.")
        df = pd.read_parquet(out)
        print(f"loaded {len(df):,} scholars, reusing existing gender")
        df = add_resolved_name(df)
    else:
        src = RESULTS / "sample_identity.parquet"
        if src.exists():
            df = pd.read_parquet(src)
        else:
            files = sorted(glob.glob(str(RESULTS / "authors_*.parquet")))
            if not files:
                raise SystemExit(f"No sample found in {RESULTS}. Run the harvest first.")
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        print(f"loaded {len(df):,} scholars")
        df = add_resolved_name(df)
        df = assign_gender(df)

    df = assign_ethnicity(df)

    pd.set_option("display.width", 200)
    report(df)

    df.to_parquet(out, index=False)
    print(f"\nWrote {out.name} ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
