"""
S1.8 -- matching engine.

Builds the comparison pairs the study actually runs on: two scholars alike
in real achievement (discipline, career stage, citations, output) AND alike
on every demographic-adjacent attribute except the one under test. Five
pair sets, each varying exactly one thing and holding the rest fixed:

  treatment_gender      -- differ on gender.     Held fixed: ethnicity, prestige (close).
  treatment_ethnicity   -- differ on ethnicity    Held fixed: gender, prestige (close).
                            (all 10 pairwise category combinations, not a
                            single reference group -- explicit design choice)
  treatment_prestige    -- differ on prestige     Held fixed: gender, ethnicity.
                            (gap must clear PRESTIGE_FAR, not just be nonzero)
  negative_control       -- differ on nothing. Same gender, ethnicity, prestige
                            (close), same laureate status. Should show no gap --
                            catches pure measurement noise.
  positive_control       -- differ on laureate status only. Same gender,
                            ethnicity, prestige (close). Should show a clear
                            gap -- from S1.7's is_laureate flag.

Prestige is deliberately NOT folded into the achievement distance -- it's
held fixed (or required to differ) via an explicit post-hoc gap check
instead, the same role the caliper already plays for achievement.

MANY-TO-MANY: a scholar appears in as many pairs as they have admissible
partners, not one. See match_two_groups. Scoring is per-scholar
(probes/scholar_scores.py), so extra pairs are free in LLM calls -- but
pairs sharing a scholar are NOT statistically independent, and the gap
table must be analysed accordingly.

Before any of that, three cleanup passes over matching_ready_data.parquet:
  1. Collapse cross-discipline duplicates to one row per scholar.
  2. Compute career age and band it.
  3. Exclude scholars whose own numbers can't be trusted for matching.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from memory_audit.data_collection.openalex import MAX_CAREER_YEARS

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # memory_audit/, parent of both data_collection/ and matching_engine/
RESULTS = PACKAGE_ROOT / "data_collection" / "results"  # reads the previous stage's output
CONFIG_PATH = ROOT / "config" / "disciplines.yaml"
MATCH_RESULTS = Path(__file__).resolve().parent / "results"  # this stage's own output

REFERENCE_YEAR = 2026  # pinned, matches eligibility.py -- reproducibility over date.today()
CAREER_BAND_WIDTH = 5  # years


def _discipline_selectors() -> dict[str, tuple[str, int]]:
    """discipline -> (selector type, id), e.g. 'physics' -> ('field', 31)."""
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {
        name: (d["selector"]["type"], d["selector"]["id"])
        for name, d in cfg["disciplines"].items()
    }


def primary_discipline(topics: list[dict], selectors: dict[str, tuple[str, int]]) -> str | None:
    """Which discipline this scholar's work actually concentrates in.

    Sums each topic's `count` into whichever of our 6 disciplines its field
    (or, for sociology, subfield) belongs to, using the exact same
    field/subfield ids the harvest itself queried by -- not a guess, the
    same selector disciplines.yaml already defines. Returns the discipline
    with the largest total, or None if nothing matches (shouldn't happen
    for a scholar the harvest itself found, but guarded rather than assumed).
    """
    totals: dict[str, int] = dict.fromkeys(selectors, 0)
    for topic in topics or []:
        count = topic.get("count", 0)
        field_id = (topic.get("field") or {}).get("id", "")
        subfield_id = (topic.get("subfield") or {}).get("id", "")
        for name, (kind, sel_id) in selectors.items():
            key = field_id if kind == "field" else subfield_id
            if key.rsplit("/", 1)[-1] == str(sel_id):
                totals[name] += count
    best = max(totals, key=totals.get)
    return best if totals[best] > 0 else None


def collapse_cross_discipline(df: pd.DataFrame) -> pd.DataFrame:
    """One row per scholar id, discipline reassigned to where their work
    actually concentrates rather than kept at an arbitrary harvested row.
    """
    import json

    single = df[~df["id"].duplicated(keep=False)].copy()
    dup_ids = df.loc[df["id"].duplicated(keep=False), "id"].unique()
    print(f"  {len(dup_ids):,} scholars appear under more than one discipline; "
          f"reassigning to their primary one")

    selectors = _discipline_selectors()
    resolved_rows = []
    unresolved = 0
    for sid in dup_ids:
        rows = df[df["id"] == sid]
        raw = rows.iloc[0]["topics"]
        topics = json.loads(raw) if isinstance(raw, str) else (raw or [])
        best = primary_discipline(topics, selectors)
        if best is None or best not in rows["discipline"].values:
            # Can't determine a primary discipline from topics, or their
            # topics don't concentrate in any discipline they were actually
            # harvested under -- keep the first harvested row rather than
            # drop the scholar entirely.
            unresolved += 1
            chosen = rows.iloc[0].copy()
        else:
            chosen = rows[rows["discipline"] == best].iloc[0].copy()
        resolved_rows.append(chosen)

    if unresolved:
        print(f"    {unresolved:,} couldn't be resolved from topics -- kept their first harvested discipline")

    resolved = pd.DataFrame(resolved_rows)
    out = pd.concat([single, resolved], ignore_index=True)
    print(f"  {len(df):,} rows -> {len(out):,} scholars")
    return out


def add_career_age(df: pd.DataFrame) -> pd.DataFrame:
    """Career age in years, and a band, for scholars whose career-start can
    be trusted. career_start_plausible already covers most bad cases (see
    eligibility.py / openalex.py's career_start()); MAX_CAREER_YEARS is a
    second, independent sanity cap in case a value slips through flagged
    True but is still absurd (nobody has a 100+ year active career).
    """
    df = df.copy()
    age = REFERENCE_YEAR - df["first_publication_year"]
    sane = df["career_start_plausible"] & age.between(0, MAX_CAREER_YEARS)
    df["career_age"] = age.where(sane)
    df["career_age_band"] = (df["career_age"] // CAREER_BAND_WIDTH * CAREER_BAND_WIDTH).where(sane)
    n_bad = (~sane).sum()
    print(f"  career age computed for {sane.sum():,} scholars "
          f"({n_bad:,} excluded: implausible or missing career-start)")
    return df


def eligible_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    """Scholars whose own numbers are trustworthy enough to be matched on.

    Excludes fragmentation candidates (their true output is understated by
    an unknown amount -- matching them on numbers known to be wrong isn't
    safe) and anyone without a usable career age. This narrows the MATCHING
    pool, not the underlying dataset -- excluded scholars are simply never
    selected as a match, nothing about them is deleted.
    """
    frag = df["fragmentation_candidate"].fillna(False).astype(bool)
    has_age = df["career_age_band"].notna()
    ok = (~frag) & has_age
    print(f"  {ok.sum():,} of {len(df):,} scholars eligible for matching "
          f"({frag.sum():,} fragmentation candidates, "
          f"{(~has_age).sum():,} no usable career age excluded)")
    return df[ok].copy()


# ------------------------------------------------------------------ pairing --

DISTANCE_FEATURES = ["log_citations", "h_index", "log_works", "career_age"]
# BUG FOUND AND FIXED 2026-08-28 -- full account in RESEARCH_LOG.md. This
# used to be Mahalanobis distance (divide by the inverse covariance of the
# standardized features), on the reasoning that citations/h-index/works are
# correlated 0.66-0.86, so treating them as independent axes would triple-
# count one underlying "how accomplished" signal. That reasoning is correct
# in the abstract, but empirically wrong for what this study actually needs:
# a user-reported case (h-index 8 vs 41, a 5.1x gap) turned out to measure
# only 0.94 Mahalanobis units -- comfortably inside caliper -- BECAUSE the
# gap showed up consistently across all three correlated features at once,
# which is exactly the direction Mahalanobis distance is designed to
# discount. Measured across all 8,630 exported pairs: distance-vs-h-index-
# ratio correlation was only 0.518, and retuning the caliper down to 1.0
# (losing half the sample) still left 5.9% of survivors with a 2x+ h-index
# gap -- the discount is structural, not a calibration problem, and no
# caliper value fixes it. Plain standardized Euclidean distance on the SAME
# features, measured on the same pairs, correlates with h-index ratio at
# 0.621 instead. Switched to that: every feature counts fully, so two
# scholars can no longer be called "close" just because their gap happens to
# lie along the pool's typical achievement-correlation direction.
#
# Standardization is per (discipline, career_age_band) stratum -- the SAME
# grouping the matching itself uses -- not global. Global standardization
# was a second, independent bug: the pool-wide std of h_index is 40.6
# (mixing early-career and legendary scholars), but e.g. the psychology/
# 25-year-career stratum's own std is only 25.2, so a genuinely large
# within-stratum gap read as a small fraction of a (wrong, too-wide) global
# SD. Verified stratum sizes before relying on this: 89 (discipline,
# career_age_band) strata total, median 156 scholars, only 10 below 30 and
# 4 below 10 -- per-stratum standardization is stable almost everywhere.
# MIN_STRATUM_FOR_STANDARDIZATION gates the few thin strata onto the
# discipline-level mean/std instead (every discipline has 2,000+ scholars,
# always stable), rather than assuming one threshold fits every stratum.
MIN_STRATUM_FOR_STANDARDIZATION = 30

# Recheck this against the real distance distribution after every matching
# change -- it was set once during design (Mahalanobis-scale) and left
# unchecked for the whole first run; that's exactly how the bug above went
# undetected. Retuned for the Euclidean metric using the real post-run
# distribution -- see RESEARCH_LOG.md for the numbers behind this value.
CALIPER = 1.5

# MATCH ON THE SCALE THE MODEL ACTUALLY READS (added 2026-08-28).
#
# The distance above works on z-scored, log-transformed features. The MODEL
# does not: prompts.py sends raw integers --
#     Publications: {works_count}
#     Total citations: {cited_by_count}
#     h-index: {h_index}
# so a pair can be "close" in z-space while showing the model two obviously
# different records. Measured on the real output: one negative_control pair
# had 360 vs 4,892 citations (13.5x) and still scored distance 1.468,
# because (a) log1p compresses a 13.5x ratio to a 2.61 gap, (b) that
# stratum's own log_citations SD is 1.782, so the gap is only 1.46 SDs, and
# (c) h-index/works/career-age were near-identical, leaving the whole
# caliper budget free for one dimension to consume. Nothing there is a bug
# -- each step is doing what it says -- but the result is a pair whose two
# prompts are not interchangeable, which is exactly what matching exists to
# prevent.
#
# So: cap the RAW ratio on the three count features the model is shown.
# career_age needs no cap -- career_age_band already exact-matches it to
# within one 5-year band (at most ~1.16x).
#
# This is an admissibility criterion, applied alongside the caliper in
# match_two_groups: a pairing that breaches it is simply not a pair.
MAX_RAW_RATIO = 2.5
RATIO_CAPPED_FEATURES = ["h_index", "cited_by_count", "works_count"]

ETHNICITY_CATEGORIES = ["nh_white", "asian", "hispanic", "mena", "nh_black"]


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardized achievement features the match distance is computed on.

    Standardized per (discipline, career_age_band) stratum -- the SAME
    grouping the matching itself uses -- falling back to the discipline
    level for strata thinner than MIN_STRATUM_FOR_STANDARDIZATION. NOT
    global: see the CALIPER comment above for why that silently broke
    sensitivity to genuinely large within-stratum achievement gaps.
    """
    df = df.copy()
    df["log_citations"] = np.log1p(df["cited_by_count"])
    df["log_works"] = np.log1p(df["works_count"])
    raw_cols = ["log_citations", "h_index", "log_works", "career_age"]

    strata_cols = ["discipline", "career_age_band"]
    strat_n = df.groupby(strata_cols, observed=True)["id"].transform("size")
    use_stratum = strat_n >= MIN_STRATUM_FOR_STANDARDIZATION

    mean = pd.DataFrame(index=df.index, columns=raw_cols, dtype=float)
    std = pd.DataFrame(index=df.index, columns=raw_cols, dtype=float)
    strat_mean = df.groupby(strata_cols, observed=True)[raw_cols].transform("mean")
    strat_std = df.groupby(strata_cols, observed=True)[raw_cols].transform("std")
    disc_mean = df.groupby("discipline", observed=True)[raw_cols].transform("mean")
    disc_std = df.groupby("discipline", observed=True)[raw_cols].transform("std")
    for col in raw_cols:
        mean[col] = strat_mean[col].where(use_stratum, disc_mean[col])
        std[col] = strat_std[col].where(use_stratum, disc_std[col])

    z = (df[raw_cols] - mean) / std
    for col in DISTANCE_FEATURES:
        df[f"z_{col}"] = z[col]
    return df


def _raw_ratio_violations(a: pd.DataFrame, b: pd.DataFrame,
                           max_ratio: float) -> np.ndarray:
    """(len(a), len(b)) boolean: True where this A-B pairing exceeds
    max_ratio on ANY of RATIO_CAPPED_FEATURES, measured on the RAW counts
    the model is shown (see the MAX_RAW_RATIO comment above).

    "+1" on both sides so a scholar with 0 of something can't divide by
    zero; at these magnitudes it doesn't move the ratio meaningfully.
    """
    bad = np.zeros((len(a), len(b)), dtype=bool)
    for col in RATIO_CAPPED_FEATURES:
        va = a[col].to_numpy(dtype=float)[:, None] + 1.0
        vb = b[col].to_numpy(dtype=float)[None, :] + 1.0
        ratio = np.maximum(va, vb) / np.minimum(va, vb)
        bad |= ratio > max_ratio
    return bad


def match_two_groups(a: pd.DataFrame, b: pd.DataFrame,
                      caliper: float = CALIPER,
                      max_ratio: float = MAX_RAW_RATIO) -> pd.DataFrame:
    """EVERY admissible A-B pairing within one stratum -- many-to-many, not
    1:1.

    Changed from optimal 1:1 assignment 2026-08-28. Under 1:1, each scholar
    got exactly one partner (the one minimising total distance across the
    stratum) and every other scholar within caliper was discarded, which
    threw away most of the admissible comparisons: measured on the real
    pool, 1:1 kept 6,747 pairs where the admissible set is 111,518. Since
    scoring is now per-SCHOLAR rather than per-pair (see
    probes/scholar_scores.py), an extra pair costs no extra LLM call -- a
    scholar is scored once and reused across every pair they appear in --
    so discarding admissible comparisons bought nothing.

    A pair is admissible iff BOTH:
      * Euclidean distance over standardized DISTANCE_FEATURES <= caliper
      * no raw-count ratio exceeds max_ratio (see _raw_ratio_violations)

    CONSEQUENCE FOR ANALYSIS: pairs sharing a scholar are not independent,
    so a plain t-test over gaps understates the standard error. The gap
    table must be analysed with scholar as a random effect (or clustered
    standard errors) -- see probes/scholar_scores.py::join_scores.
    """
    if len(a) == 0 or len(b) == 0:
        return pd.DataFrame(columns=["a_id", "b_id", "distance"])
    cols = [f"z_{c}" for c in DISTANCE_FEATURES]
    za, zb = a[cols].to_numpy(), b[cols].to_numpy()
    diff = za[:, None, :] - zb[None, :, :]  # (len(a), len(b), 4)
    cost = np.sqrt((diff ** 2).sum(axis=-1))
    admissible = (cost <= caliper) & (~_raw_ratio_violations(a, b, max_ratio))
    ai, bi = np.where(admissible)
    return pd.DataFrame({
        "a_id": a.iloc[ai]["id"].to_numpy(),
        "b_id": b.iloc[bi]["id"].to_numpy(),
        "distance": cost[ai, bi],
    })


def build_pairs(pool: pd.DataFrame, mask_a: pd.Series, mask_b: pd.Series,
                 extra_strata: tuple[str, ...] = (),
                 caliper: float = CALIPER,
                 max_ratio: float = MAX_RAW_RATIO) -> pd.DataFrame:
    """1:1 match group A against group B, stratum by stratum, so a pair
    only ever compares scholars already alike in field and career stage --
    and, via extra_strata, alike on whatever else should be held fixed
    (e.g. ("ethnicity",) so a gender-treatment pair is also same-ethnicity).
    """
    strata_cols = ["discipline", "career_age_band", *extra_strata]
    a_groups = dict(list(pool[mask_a].groupby(strata_cols, dropna=True, observed=True)))
    b_groups = dict(list(pool[mask_b].groupby(strata_cols, dropna=True, observed=True)))
    chunks = []
    for key in set(a_groups) | set(b_groups):
        a = a_groups.get(key, pool.iloc[0:0])
        b = b_groups.get(key, pool.iloc[0:0])
        pairs = match_two_groups(a, b, caliper, max_ratio)
        if len(pairs):
            key_tuple = key if isinstance(key, tuple) else (key,)
            for col, val in zip(strata_cols, key_tuple):
                pairs[col] = val
            chunks.append(pairs)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# ------------------------------------------------------------ prestige gap --

# Grounded against the real pool, not guessed: two RANDOM same-stratum
# scholars (no prestige filter at all) differ by a median of 27.8 percentile
# points (verified). PRESTIGE_CLOSE sits meaningfully under that -- close
# enough to call "comparable prestige", not just whatever chance produces.
# PRESTIGE_FAR sits meaningfully over it -- a gap clearly larger than two
# random scholars would typically show, so treatment_prestige is testing a
# real contrast, not the same noise level "close" already excludes.
PRESTIGE_CLOSE = 15.0
PRESTIGE_FAR = 40.0


def _prestige_gap(pairs: pd.DataFrame, pool: pd.DataFrame) -> pd.Series:
    lookup = pool.set_index("id")["prestige_scholar_pct"]
    return (pairs["a_id"].map(lookup) - pairs["b_id"].map(lookup)).abs()


def filter_prestige_close(pairs: pd.DataFrame, pool: pd.DataFrame,
                           threshold: float = PRESTIGE_CLOSE) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    return pairs[_prestige_gap(pairs, pool) <= threshold].copy()


def filter_prestige_far(pairs: pd.DataFrame, pool: pd.DataFrame,
                         threshold: float = PRESTIGE_FAR) -> pd.DataFrame:
    if pairs.empty:
        return pairs
    return pairs[_prestige_gap(pairs, pool) > threshold].copy()


def treatment_pairs_gender(pool: pd.DataFrame,
                            caliper: float = CALIPER) -> pd.DataFrame:
    """Differ on gender. Held fixed: ethnicity (exact match, via
    extra_strata) and prestige (close, via post-hoc filter)."""
    pairs = build_pairs(pool, pool["gender"] == "male", pool["gender"] == "female",
                         extra_strata=("ethnicity",), caliper=caliper)
    pairs = filter_prestige_close(pairs, pool)
    if len(pairs):
        pairs["pair_type"] = "treatment_gender"
    return pairs


def treatment_pairs_ethnicity(pool: pd.DataFrame,
                               caliper: float = CALIPER) -> pd.DataFrame:
    """Differ on ethnicity. Held fixed: gender (exact match) and prestige
    (close). All 10 pairwise category combinations, not a single reference
    group -- an explicit design choice (over e.g. everyone-vs-nh_white) made
    knowing the smaller categories (nh_black=519, mena=591) will likely leave
    some strata thin or empty; that's a real coverage limit to report, not a
    reason to have avoided the comparison.
    """
    from itertools import combinations
    chunks = []
    for cat_a, cat_b in combinations(ETHNICITY_CATEGORIES, 2):
        pairs = build_pairs(pool, pool["ethnicity"] == cat_a, pool["ethnicity"] == cat_b,
                             extra_strata=("gender",), caliper=caliper)
        pairs = filter_prestige_close(pairs, pool)
        if len(pairs):
            pairs["pair_type"] = f"treatment_ethnicity_{cat_a}_vs_{cat_b}"
            chunks.append(pairs)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def treatment_pairs_prestige(pool: pd.DataFrame,
                              caliper: float = CALIPER) -> pd.DataFrame:
    """Differ on prestige (gap must clear PRESTIGE_FAR). Held fixed: gender
    and ethnicity (exact match).

    Prestige has no natural two-sided split the way gender/ethnicity do, so
    one is manufactured: within each (discipline, career_age_band, gender,
    ethnicity) stratum, split scholars into above/below their OWN stratum's
    median prestige. A global median split would just reproduce discipline-
    level prestige differences instead of a clean within-stratum contrast.
    Achievement-matching then runs across that split exactly like any other
    treatment pair, and the FAR filter keeps only the pairs whose actual
    gap is clearly bigger than chance, not just technically on opposite
    sides of a median.
    """
    has_prestige = pool["prestige_scholar_pct"].notna()
    side_a = pd.Series(False, index=pool.index)
    strata_cols = ["discipline", "career_age_band", "gender", "ethnicity"]
    for _, g in pool[has_prestige].groupby(strata_cols, dropna=True, observed=True):
        if len(g) < 2:
            continue
        med = g["prestige_scholar_pct"].median()
        side_a.loc[g.index[g["prestige_scholar_pct"] > med]] = True

    pairs = build_pairs(pool, side_a & has_prestige, ~side_a & has_prestige,
                         extra_strata=("gender", "ethnicity"), caliper=caliper)
    pairs = filter_prestige_far(pairs, pool)
    if len(pairs):
        pairs["pair_type"] = "treatment_prestige"
    return pairs


def negative_control_pairs(pool: pd.DataFrame, seed: int = 42,
                            caliper: float = CALIPER) -> pd.DataFrame:
    """Differ on nothing. Same gender, ethnicity, laureate status (all exact
    match, via extra_strata) and close prestige -- should show no gap.

    Each (gender, ethnicity, is_laureate) subgroup is randomly split in
    half; build_pairs then matches the two halves exactly like a treatment
    comparison. Nothing demographic or reputational differs between a pair
    by construction, so any gap the probes show here is pure measurement
    noise, not a real effect.
    """
    rng = np.random.default_rng(seed)
    side_a = pd.Series(False, index=pool.index)
    for gender in ["male", "female"]:
        for eth in ETHNICITY_CATEGORIES:
            for laureate in [True, False]:
                idx = pool.index[(pool["gender"] == gender) & (pool["ethnicity"] == eth)
                                  & (pool["is_laureate"] == laureate)]
                if len(idx) < 2:
                    continue
                chosen = rng.choice(idx, size=len(idx) // 2, replace=False)
                side_a.loc[chosen] = True
    valid = pool["gender"].isin(["male", "female"]) & pool["ethnicity"].isin(ETHNICITY_CATEGORIES)
    pairs = build_pairs(pool, side_a & valid, ~side_a & valid,
                         extra_strata=("gender", "ethnicity", "is_laureate"),
                         caliper=caliper)
    pairs = filter_prestige_close(pairs, pool)
    if len(pairs):
        pairs["pair_type"] = "negative_control"
    return pairs


def positive_control_pairs(pool: pd.DataFrame,
                            caliper: float = CALIPER) -> pd.DataFrame:
    """Differ on laureate status only. Held fixed: gender, ethnicity (exact
    match) and prestige (close)."""
    pairs = build_pairs(pool, pool["is_laureate"] == True, pool["is_laureate"] == False,  # noqa: E712
                         extra_strata=("gender", "ethnicity"), caliper=caliper)
    pairs = filter_prestige_close(pairs, pool)
    if len(pairs):
        pairs["pair_type"] = "positive_control"
    return pairs


def report(all_pairs: pd.DataFrame, pool: pd.DataFrame | None = None) -> None:
    print(f"\n{len(all_pairs):,} total pairs\n")
    print("--- by pair type ---")
    print(all_pairs["pair_type"].value_counts().to_string())
    print("\n--- distance distribution (lower = closer match) ---")
    print(all_pairs.groupby("pair_type")["distance"].describe()[["mean", "50%", "max"]].round(3).to_string())

    if pool is None:
        return

    # Audit on RAW, untransformed numbers -- see section 4.15 of RESEARCH_LOG.md.
    # The bug this guards against was completely invisible in z-space (every
    # distance looked small and in-caliper) and only showed up when the raw
    # counts were compared directly. Checking the same transformed space the
    # matcher optimises in cannot catch a flaw in that space's own assumptions.
    print("\n--- worst RAW ratio per pair type (guard against 4.15 recurring) ---")
    lookup = pool.set_index("id")
    worst = None
    for col in RATIO_CAPPED_FEATURES:
        va = all_pairs["a_id"].map(lookup[col]).to_numpy(dtype=float) + 1.0
        vb = all_pairs["b_id"].map(lookup[col]).to_numpy(dtype=float) + 1.0
        ratio = np.maximum(va, vb) / np.minimum(va, vb)
        worst = ratio if worst is None else np.maximum(worst, ratio)
    audit = all_pairs.assign(worst_raw_ratio=worst)
    print(audit.groupby("pair_type")["worst_raw_ratio"]
          .agg(["mean", "max"]).round(2).to_string())

    violations = int((audit["worst_raw_ratio"] > MAX_RAW_RATIO + 1e-9).sum())
    verdict = "OK" if violations == 0 else f"*** {violations} VIOLATIONS ***"
    print(f"\n  cap = {MAX_RAW_RATIO}x on {', '.join(RATIO_CAPPED_FEATURES)}: {verdict}")

    # Many-to-many reuses each scholar across pairs, so the two numbers that
    # actually drive cost and analysis are: how many DISTINCT scholars need a
    # score (= LLM calls), and how unevenly pairs are spread across them
    # (= how much a mixed-effects model has to do).
    everyone = pd.concat([all_pairs["a_id"], all_pairs["b_id"]])
    per_scholar = everyone.value_counts()
    print("\n--- many-to-many shape ---")
    print(f"  distinct scholars used     {per_scholar.size:,}  <- LLM calls, not pair count")
    print(f"  pairs per scholar          mean {per_scholar.mean():.1f}   "
          f"median {per_scholar.median():.0f}   max {per_scholar.max():,}")
    print(f"  pairs per call             {len(all_pairs) / max(per_scholar.size, 1):.1f}x")


# current_inst was MISSING here until 2026-08-28, and its absence was silent:
# prompts.build_prompt reads row.get(f"{side}_current_inst"), which returned
# None, which pd.notna() treats as "no institution to show", so the prestige
# arm rendered NO institution line despite being run with
# include_institution=True -- i.e. the one arm whose treatment IS the
# institution was measuring nothing. No error was raised at any point.
ENRICH_COLS = ["resolved_name", "discipline", "gender", "ethnicity", "h_index",
               "cited_by_count", "works_count", "career_age", "prestige_scholar_pct",
               "is_laureate", "prize_name", "current_inst"]


def enrich_pairs(pairs: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Attach scholar details for both sides of a pair -- a bare id/distance
    table isn't reviewable on its own, and every other CSV this pipeline has
    produced has carried enough columns to inspect directly.
    """
    lookup = pool.set_index("id")
    out = pairs.copy()
    for side, id_col in [("a", "a_id"), ("b", "b_id")]:
        for col in ENRICH_COLS:
            out[f"{side}_{col}"] = out[id_col].map(lookup[col])
    return out


def export_pair_type_csvs(all_pairs: pd.DataFrame, pool: pd.DataFrame,
                          out_dir: Path = MATCH_RESULTS,
                          caliper: float = CALIPER) -> None:
    """One CSV per pair_type in `out_dir`, replacing whatever a previous run
    left THERE -- files from an old distance formula or caliper shouldn't sit
    next to new ones under the same names.

    out_dir is a parameter so a run at a different caliper can be written
    beside the default set instead of over it: the clear-before-write above
    is scoped to the directory being written, so
    `--out-dir results_c03` leaves results/ untouched. Every row also carries
    the caliper it was built with, so a CSV can never be misattributed to the
    wrong run once it has been copied somewhere.
    """
    if out_dir.exists():
        removed = [f for f in out_dir.iterdir() if f.is_file()]
        for f in removed:
            f.unlink()
        if removed:
            print(f"  removed {len(removed)} file(s) from a previous run: "
                  f"{', '.join(f.name for f in removed)}")
    else:
        out_dir.mkdir(parents=True)

    enriched = enrich_pairs(all_pairs, pool)
    enriched["caliper"] = caliper
    for pair_type, group in enriched.groupby("pair_type"):
        out = out_dir / f"{pair_type}.csv"
        group.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  wrote {out.name} ({len(group):,} pairs)")


def main(caliper: float = CALIPER, out_dir: Path = MATCH_RESULTS) -> None:
    src = RESULTS / "matching_ready_data.parquet"
    df = pd.read_parquet(src)
    print(f"loaded {len(df):,} rows")
    print(f"caliper = {caliper}   ->  {out_dir}\n")

    print("collapsing cross-discipline duplicates:")
    df = collapse_cross_discipline(df)

    print("\ncomputing career age:")
    df = add_career_age(df)

    print("\nfiltering to the matching-eligible pool:")
    pool = eligible_for_matching(df)
    pool = add_distance_features(pool)

    print("\nbuilding pairs:")
    all_pairs = pd.concat([
        treatment_pairs_gender(pool, caliper=caliper),
        treatment_pairs_ethnicity(pool, caliper=caliper),
        treatment_pairs_prestige(pool, caliper=caliper),
        negative_control_pairs(pool, caliper=caliper),
        positive_control_pairs(pool, caliper=caliper),
    ], ignore_index=True)

    report(all_pairs, pool)

    # Only the default run owns the legacy parquet; a side-by-side run at a
    # different caliper must not touch the previous stage's artefacts.
    if out_dir == MATCH_RESULTS:
        old = RESULTS / "matched_pairs.parquet"
        if old.exists():
            old.unlink()
            print(f"\ndeleted stale {old.name} (superseded by per-pair-type CSVs)")

    print("\nwriting one CSV per pair type:")
    export_pair_type_csvs(all_pairs, pool, out_dir=out_dir, caliper=caliper)


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--caliper", type=float, default=CALIPER,
                    help=f"max standardized Euclidean distance (default {CALIPER})")
    ap.add_argument("--out-dir", default=None,
                    help="output dir name under matching_engine/ "
                         "(default: results). Use a separate name to keep an "
                         "existing pair set instead of overwriting it.")
    a = ap.parse_args()
    out = MATCH_RESULTS if a.out_dir is None else MATCH_RESULTS.parent / a.out_dir
    main(caliper=a.caliper, out_dir=out)


if __name__ == "__main__":
    _cli()
