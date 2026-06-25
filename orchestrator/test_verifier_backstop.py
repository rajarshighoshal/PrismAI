"""Offline unit tests for the verbatim backstop — the false-positive guard on the can't-lie
guarantee. `_claim_verbatim_in_source` rescues a flagged phrase ONLY when it appears
near-verbatim and contiguous in the source (so the auditor can't strip a real fact it
merely mis-flagged), and must NEVER rescue a semantic inflation. No API, no network.

  python -m orchestrator.test_verifier_backstop
"""
from orchestrator.verifier import _norm_token_str, _claim_verbatim_in_source

_fails = []


def check(name, cond):
    print(f"{'PASS' if cond else 'FAIL'}: {name}")
    if not cond:
        _fails.append(name)


def _in(phrase, source):
    return _claim_verbatim_in_source(phrase, _norm_token_str(source))


# ── normalization: case-folded, punctuation-stripped, whole tokens, space-wrapped ──
check("norm: lowercases + strips punctuation to whole tokens",
      _norm_token_str("Led the Data-Team!") == " led the data team ")
check("norm: collapses runs of whitespace", _norm_token_str("a   b\n c") == " a b c ")

# ── rescue a REAL verbatim quote the auditor mis-flagged ──
check("rescue: contiguous multi-token phrase present in source",
      _in("democratize logistics for small businesses",
          "Our mission is to democratize logistics for small businesses."))
check("rescue: case/punctuation-insensitive verbatim quote",
      _in("Brewing happiness since 1998", "Tagline: brewing happiness, since 1998!"))

# ── never rescue an inflation / non-verbatim claim ──
check("no-rescue: 'led the data team' when the source says 'collaborated with'",
      not _in("led the data team", "collaborated with the data team"))
check("no-rescue: tokens present but not contiguous in source",
      not _in("fraud model at Acme", "Acme built a churn model and a fraud model"))

# ── whole-token boundaries: never a raw substring match ──
check("boundary: 'led team' does not match inside 'sled team'",
      not _in("led team", "the sled team raced"))
check("boundary: '12 k' does not match inside '120 k'",
      not _in("12 k", "the cost was 120 k per year"))

# ── too-short phrases defer to the auditor (never rescued mechanically) ──
check("short: a single-token phrase is never rescued",
      not _in("Stanford", "she studied at Stanford"))
check("short: an empty phrase returns False", not _in("", "anything at all"))

print(f"\n{'all verbatim-backstop tests passed' if not _fails else f'{len(_fails)} FAILED: {_fails}'}")
import sys
sys.exit(1 if _fails else 0)
