"""
Recover full first names from OpenAlex name variants.

OpenAlex's `display_name` is often initials-only ("M. Meyerson"), which is fatal
for gender inference: 72.8% of initials-only names resolve to unknown gender,
against 3.4% of full names. Physics is 63.6% initials-only, so it alone loses
half its gender data.

But `display_name_alternatives` usually carries the full form somewhere:

    display_name : M. Meyerson
    alternatives : ['M. Meyerson', 'Matthew Meyerson', 'Meyerson, M.', ...]

Three shapes have to be handled: "First Last", "Last, First", and shouted
surnames ("Yoshihiho FUKUDA").

The risk this guards against
----------------------------
An alternatives list is only as clean as the author record, and merged records
carry other people's names. P. L. Bhatnagar (the physicist of BGK-equation
fame, died 1976) lists "Pankhuri Bhatnagar" as an alternative -- a different
person, same surname, same first initial. Accepting that would not just be
wrong, it would flip the gender, corrupting the exact variable this exists to
improve.

So expansion is deliberately conservative:

  * surname must match after normalising case, accents and punctuation
  * the full first name must start with the same letter as the original initial
  * if two candidates propose DIFFERENT first names, the record is ambiguous
    and is left alone rather than guessed

Bhatnagar still slips through the first two checks -- one candidate, matching
initial -- which is why every expansion is recorded in its own column rather
than silently overwriting `display_name`.
"""

from __future__ import annotations

import re
import unicodedata

_INITIAL = re.compile(r"^[A-Za-z]\.?$")
_PUNCT = re.compile(r"[^\w\s]")


def fold(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(_PUNCT.sub(" ", stripped).lower().split())


def is_initial(token: str) -> bool:
    """True for an initial or a run of them: 'M', 'M.', 'JN', 'RR'.

    The multi-letter case matters: OpenAlex lists "JN Hirschhorn" beside
    "Hirschhorn, Joel N.". Reading "JN" as a first name makes the record look
    like two people disagreeing, and a genuinely recoverable name is thrown
    away as ambiguous. All-caps runs of up to three letters are initials;
    longer all-caps tokens are shouted surnames ("FUKUDA") and are left alone.
    """
    token = str(token).strip()
    if _INITIAL.match(token):
        return True
    bare = token.replace(".", "")
    return bare.isalpha() and bare.isupper() and len(bare) <= 3


def normalise_variant(variant: str) -> str:
    """Put a name variant into 'First Last' order.

    OpenAlex mixes 'Joel N. Hirschhorn' and 'Hirschhorn, Joel N.' in the same
    list, so the comma form is flipped before anything else looks at it.
    """
    variant = str(variant).strip()
    if "," in variant:
        last, _, first = variant.partition(",")
        variant = f"{first.strip()} {last.strip()}".strip()
    return " ".join(variant.split())


def surname_of(name: str) -> str:
    """Everything after the first token, minus middle initials.

    Keeps compound surnames intact ('de Krijger', 'van der Berg') by dropping
    only the standalone initials rather than taking the last token.
    """
    tokens = str(name).split()
    if len(tokens) < 2:
        return ""
    rest = [t for t in tokens[1:] if not is_initial(t)]
    return " ".join(rest)


def first_token(name: str) -> str:
    tokens = str(name).split()
    return tokens[0] if tokens else ""


def _titlecase_if_shouted(name: str) -> str:
    """'Yoshihiho FUKUDA' -> 'Yoshihiho Fukuda'; leave mixed case alone."""
    return " ".join(
        tok.capitalize() if tok.isupper() and len(tok) > 1 else tok
        for tok in str(name).split()
    )


def resolve_full_name(display_name: str, alternatives) -> tuple[str, str]:
    """Return (best_name, status).

    status is one of:
        already_full  -- display_name had a real first name; unchanged
        expanded      -- an alternative supplied the first name
        ambiguous     -- candidates disagreed on the first name; unchanged
        no_candidate  -- nothing usable in the alternatives; unchanged
    """
    display_name = str(display_name).strip()
    if not is_initial(first_token(display_name)):
        return display_name, "already_full"

    initial = first_token(display_name).rstrip(".").lower()
    target_surname = fold(surname_of(display_name))
    if not target_surname:
        return display_name, "no_candidate"

    candidates: dict[str, str] = {}
    for raw in alternatives or []:
        # Test the head for initial-ness BEFORE de-shouting: "JN Hirschhorn"
        # title-cased to "Jn" is no longer all-caps, so the initial-cluster
        # check would miss it and the record would look ambiguous.
        variant = normalise_variant(raw)
        head = first_token(variant)
        if not head or is_initial(head):
            continue
        variant = _titlecase_if_shouted(variant)
        head = first_token(variant)
        if fold(surname_of(variant)) != target_surname:
            continue
        if head[:1].lower() != initial:
            continue
        # Key on the first name so differently-punctuated repeats of the same
        # person collapse, but two genuinely different first names do not.
        candidates.setdefault(fold(head), variant)

    if not candidates:
        return display_name, "no_candidate"
    if len(candidates) > 1:
        return display_name, "ambiguous"

    return next(iter(candidates.values())), "expanded"


if __name__ == "__main__":
    _ALREADY_FULL = "Anne LHuillier"
    cases = [
        ("M. Meyerson",
         ["M. Meyerson", "Matthew Meyerson", "Meyerson, M.", "Meyerson, Matthew"],
         "Matthew Meyerson", "expanded"),
        ("J. N. Hirschhorn",
         ["Hirschhorn, J. N.", "Hirschhorn, Joel N.", "JN Hirschhorn"],
         "Joel N. Hirschhorn", "expanded"),
        ("Y Fukuda",
         ["Y Fukuda", "Y. Fukuda", "Yoshihiho FUKUDA"],
         "Yoshihiho Fukuda", "expanded"),
        ("R. De Krijger",
         ["R de Krijger", "RR De Krijger", "Ronald de Krijger"],
         "Ronald de Krijger", "expanded"),
        # surname must match: 'Becerra Ortiz' is not 'Ortiz'
        ("P. Ortiz",
         ["Ortiz, P", "P Ortiz", "P. Becerra Ortiz"],
         "P. Ortiz", "no_candidate"),
        # "Samir" is filtered by the initial check, "Gardabbou Khaled" by the
        # surname check, leaving exactly one candidate -- the right one.
        ("K. Gardabbou",
         ["Khaled Gardabbou", "Gardabbou Khaled", "Samir Gardabbou"],
         "Khaled Gardabbou", "expanded"),
        # two real first names for one initial: refuse to guess
        ("A. Smith",
         ["Alice Smith", "Andrew Smith"],
         "A. Smith", "ambiguous"),
        (_ALREADY_FULL, [_ALREADY_FULL], _ALREADY_FULL, "already_full"),
    ]
    ok = True
    print(f"{'input':22s} {'->':2s} {'result':24s} status")
    print("-" * 70)
    for display, alts, want, want_status in cases:
        got, status = resolve_full_name(display, alts)
        good = status == want_status and (want is None or got == want)
        ok &= good
        print(f"{display:22s} -> {got:24s} {status:13s} {'OK' if good else 'FAIL'}")
    print()
    print("all pass" if ok else "SOME FAILED")
