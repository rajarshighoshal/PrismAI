"""Time context for the model: what 'now' is, and resume-after-gap awareness."""
from datetime import datetime, timezone, timedelta

from . import config


def _gap_note(last_active) -> str:
    """One line telling the model the user is RESUMING after a gap — but only when the
    previous message was a prior calendar day or more (in local time). Same-day follow-ups
    get nothing, so a continuous session stays clean. Fixes 'replies days later, acts like
    no time passed' without timestamping every turn."""
    if not last_active:
        return ""
    try:
        tz = timezone(timedelta(minutes=config.LOCAL_TZ_OFFSET_MINUTES))
        prev = datetime.fromtimestamp(float(last_active), tz)
        days = (datetime.now(tz).date() - prev.date()).days
    except Exception:
        return ""
    if days < 1:
        return ""
    when = "yesterday" if days == 1 else f"{days} days ago"
    return (f"NOTE: the user is resuming this conversation after a gap — their previous "
            f"message was {when} ({prev:%A, %d %B %Y}), not today. Earlier turns are older; "
            f"don't treat the conversation as one continuous sitting.")


def _now_line() -> str:
    """Tell the model what 'today' is, in the USER's local time — otherwise it has no idea
    and burns tokens debating whether a date (e.g. 'finished my MS in May 2026') is past or
    future. The OpenAI-style request OWUI sends carries no timezone, so we format in a
    configured local offset (LOCAL_TZ_OFFSET_MINUTES) rather than the server's UTC clock."""
    tz = timezone(timedelta(minutes=config.LOCAL_TZ_OFFSET_MINUTES))
    now = datetime.now(tz)
    # Both date orders, so a letterhead in either style ("10 June 2026" / "June 10, 2026")
    # is a contiguous verbatim span of the grounding source — the backstop then protects a
    # dated letterhead mechanically regardless of which format the writer picks.
    return (f"The current date and time is {now:%A, %d %B %Y} ({now:%B %d, %Y}), "
            f"{now:%H:%M} {config.LOCAL_TZ_LABEL}. Treat this as 'now' when reasoning "
            "about whether any date or time is in the past or future.")
