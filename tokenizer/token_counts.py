"""
S1.6 -- per-tokeniser token counts for scholar names.

Why: the completion-probability probe scores a name token-by-token, so a
name that fragments into more tokens is mechanically harder to score high
on, independent of real fame. Transliterated/non-Western names fragment
more on average even in identical Latin script (verified earlier: "Saman
Esmaeilnejad" -> 8 pieces vs "Alex Bridgland" -> 4 on Pythia's tokenizer) --
left uncontrolled, that reads as ethnicity bias when it's a tokenizer
artifact. Token count is used twice downstream, not here: to normalise
each probe score per token as it's computed, and again as a regression
covariate in the final analysis. It is NOT a matching variable (S1.8
doesn't touch it) -- normalisation and regression control handle this
confound, matching handles achievement/demographics separately.

Covers the 8 open-weight families that carry the log-probability probes
(hosted models are recognition-only, no logprobs, so this doesn't apply to
them). One tokenizer per family, not per checkpoint size -- verified
earlier that all 8, including the two that were gated (Llama, Gemma),
download with HF_TOKEN in place.

Every name is tokenized WITH a leading space (" " + name), not bare --
BPE tokenizers split a word differently depending on whether it's preceded
by whitespace, and a name almost always appears preceded by a space in a
real prompt. Tokenizing bare would silently measure the wrong thing.

Output is WIDE: one row per scholar, one column per tokenizer family (8
token-count columns total), plus id/orcid/resolved_name as identifiers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import memory_audit  # noqa: F401 -- triggers .env loading (HF_TOKEN) on import

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "data_collection" / "results" / "matching_ready_data.parquet"
RESULTS = Path(__file__).resolve().parent / "results"

MODELS = {
    "pythia": "EleutherAI/pythia-410m",
    "olmo": "allenai/OLMo-2-1124-7B",
    "qwen": "Qwen/Qwen2.5-7B",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "mistral": "mistralai/Mistral-7B-v0.3",
    "llama": "meta-llama/Llama-3.1-8B",
    "gemma": "google/gemma-2-9b",
}


def load_scholar_names(path: Path = SRC) -> pd.DataFrame:
    """One row per unique scholar: id, orcid (as stored -- full ORCID URL,
    not stripped), resolved_name. Token count depends only on the name
    string, not discipline, so this just dedupes on id -- no need for
    matching_engine's primary-discipline logic here.
    """
    df = pd.read_parquet(path, columns=["id", "orcid", "resolved_name"])
    before = len(df)
    df = df.drop_duplicates(subset="id").reset_index(drop=True)
    print(f"  {before:,} rows -> {len(df):,} unique scholars "
          f"({df['orcid'].notna().sum():,} with an ORCID)")
    return df


def add_token_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds one column per family: {family}_tokens."""
    df = df.copy()
    for family, model_id in MODELS.items():
        print(f"  tokenizing with {family} ({model_id})...")
        if family == "deepseek":
            # AutoTokenizer loads DeepSeek as LlamaTokenizerFast, which in
            # transformers v5.x hardcodes a Metaspace pre-tokenizer that
            # silently overrides tokenizer.json's real ByteLevel setting
            # (huggingface/transformers#45488). Confirmed on our own data:
            # returns 0 tokens on Cyrillic/Greek names, and even silently
            # drops the leading space on Latin ones ("Alex" instead of
            # "_Alex"). PreTrainedTokenizerFast bypasses the override and
            # reads tokenizer.json as actually published.
            tok = PreTrainedTokenizerFast.from_pretrained(model_id)
        else:
            tok = AutoTokenizer.from_pretrained(model_id)
        df[f"{family}_tokens"] = [len(tok.encode(" " + str(n))) for n in df["resolved_name"]]
    return df


def report(df: pd.DataFrame) -> None:
    cols = [f"{f}_tokens" for f in MODELS]
    print(f"\n{len(df):,} scholars x {len(MODELS)} families\n")
    print("--- token count by family ---")
    print(df[cols].agg(["mean", "median", "min", "max"]).round(2).to_string())


def main() -> None:
    print("loading scholar names:")
    df = load_scholar_names()

    print()
    df = add_token_counts(df)

    report(df)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_parquet = RESULTS / "token_counts.parquet"
    df.to_parquet(out_parquet, index=False)
    out_csv = RESULTS / "token_counts.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nWrote {out_parquet.name} and {out_csv.name} ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
