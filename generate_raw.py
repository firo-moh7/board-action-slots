#!/usr/bin/env python3
"""Generate synthetic Board Action Card raw minutes (CC0-style).

Design goals for Eris acceptance:
- Naive classic-verb / last-match heuristics score low (~0.2–0.4)
- Structure-aware regex should NOT saturate (~0.55–0.70)
- DistilBERT FT can learn discourse binding (~0.72–0.85)
- Soft ceiling from ambiguity + post traps (~0.88–0.92)
"""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

import pandas as pd

SEED = 20261009
N_ROWS = 1200
OUT = Path(__file__).resolve().parent / "raw" / "minutes.csv"

STATUSES = ("approved", "deferred", "rejected", "noted_only")
DUE_BUCKETS = ("this_week", "this_month", "this_quarter", "undated")

FAMILIES = (
    "formal_board",
    "scrum_standup",
    "email_recap",
    "risk_committee",
    "product_council",
    "facilities_ops",
    "finance_close",
    "legal_review",
    "hiring_panel",
    "vendor_steerco",
    "incident_warroom",
    "academic_dept",
)

FIRST = (
    "Jordan", "Alex", "Sam", "Riley", "Casey", "Morgan", "Quinn", "Avery",
    "Taylor", "Jamie", "Drew", "Cameron", "Reese", "Parker", "Skyler", "Dana",
)
LAST = (
    "Nguyen", "Patel", "Brooks", "Okoye", "Silva", "Berg", "Khan", "Ortiz",
    "Iwata", "Keller", "Diaz", "Singh", "Cohen", "Novak", "Hassan", "Floyd",
)

SUBJECTS = (
    "vendor escrow renewal",
    "warehouse HVAC retrofit",
    "Q4 hiring freeze exception",
    "api rate-limit increase",
    "disaster recovery drill",
    "office lease extension",
    "customer refund policy rewrite",
    "nightly backup window change",
    "contractor badge access",
    "marketing spend reallocation",
    "on-call rotation redesign",
    "legacy database migration",
    "safety audit remediation",
    "partner SOW amendment",
    "inventory write-down review",
    "remote stipend adjustment",
    "encryption key rotation",
    "pilot branch opening",
    "sla credit schedule",
    "training budget uplift",
)

# Gold decisions — soft lexicon; also mirrored in SOFT_TRAPS so token presence ≠ label.
STATUS_DECISION = {
    "approved": (
        "The chair recorded assent on {subject}.",
        "Assent was entered for {subject}.",
        "Final card: assent — {subject}.",
        "Card outcome for {subject}: assent.",
        "Assent stands for {subject}.",
        "Official mark on {subject} is assent.",
    ),
    "deferred": (
        "The chair parked {subject} for a later docket.",
        "Parking was entered for {subject}.",
        "Final card: park — {subject}.",
        "Card outcome for {subject}: park.",
        "Parking stands for {subject}.",
        "Official mark on {subject} is park.",
    ),
    "rejected": (
        "The chair recorded a denial on {subject}.",
        "Denial was entered for {subject}.",
        "Final card: denial — {subject}.",
        "Card outcome for {subject}: denial.",
        "Denial stands for {subject}.",
        "Official mark on {subject} is denial.",
    ),
    "noted_only": (
        "The chair filed {subject} as information-only.",
        "An information-only entry was made for {subject}.",
        "Final card: information-only — {subject}.",
        "Card outcome for {subject}: information-only.",
        "Information-only stands for {subject}.",
        "Official mark on {subject} is information-only.",
    ),
}

SOFT_TRAPS = {
    "approved": (
        "Draft language floated assent on {subject}, but no card was written then.",
        "A mock final card reading assent — {subject} was shown as an example only.",
    ),
    "deferred": (
        "Draft language floated parking {subject}, but no card was written then.",
        "A mock final card reading park — {subject} was shown as an example only.",
    ),
    "rejected": (
        "Draft language floated denial on {subject}, but no card was written then.",
        "A mock final card reading denial — {subject} was shown as an example only.",
    ),
    "noted_only": (
        "Draft language floated information-only for {subject}, but no card was written then.",
        "A mock final card reading information-only — {subject} was shown as an example only.",
    ),
}

STATUS_TRAPS = {
    "approved": (
        "In debate, {speaker} said \"we approved {subject} in spirit,\" which is not a vote.",
        "A prior cycle already approved a cousin of {subject}; that history was only context.",
        "The packet's sample language still reads \"approved: {subject}\" from an old template.",
    ),
    "deferred": (
        "In debate, {speaker} said \"we deferred {subject} last time,\" which is not today's act.",
        "A prior cycle deferred a cousin of {subject}; that history was only context.",
        "The packet's sample language still reads \"deferred: {subject}\" from an old template.",
    ),
    "rejected": (
        "In debate, {speaker} said \"we rejected {subject} before,\" which is not today's act.",
        "A prior cycle rejected a cousin of {subject}; that history was only context.",
        "The packet's sample language still reads \"rejected: {subject}\" from an old template.",
    ),
    "noted_only": (
        "In debate, {speaker} said \"we noted {subject} only,\" which is not today's act.",
        "A prior cycle noted a cousin of {subject} with no action; that history was only context.",
        "The packet's sample language still reads \"noted only: {subject}\" from an old template.",
    ),
}

# Prefer less template-y gold dues; common "Follow-through target:" lines are mostly traps.
DUE_GOLD = {
    "this_week": (
        "Cadence chosen: week-scale.",
        "Clocked for the near horizon.",
        "Time box: near.",
        "Target window set to this week.",
    ),
    "this_month": (
        "Cadence chosen: month-scale.",
        "Clocked for the mid horizon.",
        "Time box: mid.",
        "Target window set to this month.",
    ),
    "this_quarter": (
        "Cadence chosen: quarter-scale.",
        "Clocked for the far horizon.",
        "Time box: far.",
        "Target window set to this quarter.",
    ),
    "undated": (
        "Cadence chosen: open-scale.",
        "Clocked for an open horizon.",
        "Time box: open.",
        "Target window set to none.",
    ),
}

DUE_TRAP_LINES = (
    "A sticky note still says follow-through target: this week for a closed item.",
    "A sticky note still says follow-through target: this month for a closed item.",
    "A sticky note still says follow-through target: this quarter for a closed item.",
    "A sticky note still says follow-through target: none for a closed item.",
    "Someone pasted schedule band = this week for a different ticket.",
    "Someone pasted schedule band = this month for a different ticket.",
    "Someone pasted target window set to this quarter for a different ticket.",
    "Cadence chosen: week-scale appears in an appendix for another chore.",
    "Cadence chosen: month-scale appears in an appendix for another chore.",
    "Sidebar chatter kept saying due before Friday about unrelated chores.",
)

# Gold avoids the most common trap phrasing ("will own the follow-up").
OWNER_GOLD = (
    "Follow-up owner: {owner}.",
    "Assign follow-up to {owner}.",
    "Owner for follow-up is {owner}.",
    "Follow-up sits with {owner}.",
)

FAMILY_OPENERS = {
    "formal_board": "Board minutes.",
    "scrum_standup": "Standup notes.",
    "email_recap": "Recap email body.",
    "risk_committee": "Risk committee log.",
    "product_council": "Product council notes.",
    "facilities_ops": "Facilities ops brief.",
    "finance_close": "Finance close huddle.",
    "legal_review": "Legal review minute.",
    "hiring_panel": "Hiring panel note.",
    "vendor_steerco": "Vendor steerco minute.",
    "incident_warroom": "Incident war-room log.",
    "academic_dept": "Department meeting note.",
}

FAMILY_SPEAKER = {
    "formal_board": "{speaker} raised a point of order during discussion.",
    "scrum_standup": "{speaker} flagged a blocker in the round-robin.",
    "email_recap": "{speaker} summarized prior thread noise.",
    "risk_committee": "{speaker} walked the residual-risk slide.",
    "product_council": "{speaker} replayed the customer quote.",
    "facilities_ops": "{speaker} reported a site walk-through.",
    "finance_close": "{speaker} cited the variance bridge.",
    "legal_review": "{speaker} flagged a precedent citation.",
    "hiring_panel": "{speaker} compared rubric scores.",
    "vendor_steerco": "{speaker} reviewed the scorecard.",
    "incident_warroom": "{speaker} restated the timeline.",
    "academic_dept": "{speaker} referenced the curriculum memo.",
}


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _span_at(text: str, surface: str, start: int) -> str:
    end = start + len(surface)
    assert text[start:end] == surface
    return f"{surface}@{start}:{end}"


def _anchor(text: str) -> str:
    ch = text[0] if text else " "
    return f"{ch}@0:1"


def _normalize_subject(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s&-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _pick_status(rng: random.Random) -> str:
    return rng.choices(STATUSES, weights=[0.24, 0.28, 0.20, 0.28], k=1)[0]


def _pick_due(rng: random.Random, status: str) -> str:
    if status == "noted_only" and rng.random() < 0.55:
        return "undated"
    if status == "rejected" and rng.random() < 0.35:
        return "undated"
    return rng.choice(DUE_BUCKETS)


def build_row(rng: random.Random, idx: int) -> dict:
    family = rng.choice(FAMILIES)
    status = _pick_status(rng)
    due = _pick_due(rng, status)
    subject = rng.choice(SUBJECTS)
    speaker = _name(rng)

    if rng.random() < 0.22 or (status == "noted_only" and rng.random() < 0.45):
        owner = "unassigned"
    else:
        owner = _name(rng)
        while owner == speaker:
            owner = _name(rng)

    trap_statuses = [s for s in STATUSES if s != status]
    rng.shuffle(trap_statuses)

    classic_traps = [
        rng.choice(STATUS_TRAPS[trap_statuses[0]]).format(subject=subject, speaker=speaker)
    ]
    if rng.random() < 0.75:
        classic_traps.append(
            rng.choice(STATUS_TRAPS[trap_statuses[1]]).format(subject=subject, speaker=speaker)
        )
    soft_traps = [
        rng.choice(SOFT_TRAPS[trap_statuses[0]]).format(subject=subject)
    ]
    if rng.random() < 0.55:
        soft_traps.append(rng.choice(SOFT_TRAPS[status]).format(subject=subject))

    decision = rng.choice(STATUS_DECISION[status]).format(subject=subject)
    due_line = rng.choice(DUE_GOLD[due])
    if owner == "unassigned":
        owner_clause = "No individual owner was recorded for the follow-up."
        owner_surface = None
    else:
        owner_clause = rng.choice(OWNER_GOLD).format(owner=owner)
        owner_surface = owner

    opener = FAMILY_OPENERS[family]
    speaker_line = FAMILY_SPEAKER[family].format(speaker=speaker)
    speaker_own = f"{speaker} will own the briefing deck."
    other_subject = rng.choice([s for s in SUBJECTS if s != subject])
    early_due = rng.choice(DUE_TRAP_LINES)
    decoy_owner = _name(rng)
    while decoy_owner in {speaker, owner}:
        decoy_owner = _name(rng)

    body = [opener, speaker_line, classic_traps[0], soft_traps[0], early_due, speaker_own]
    if len(classic_traps) > 1:
        body.append(classic_traps[1])
    if len(soft_traps) > 1:
        body.append(soft_traps[1])
    if rng.random() < 0.65:
        body.append(f"Parking-lot chatter also mentioned {other_subject} without a card.")

    # Shuffle card block order so "due immediately before owner" is unreliable.
    card = [decision, due_line, owner_clause]
    rng.shuffle(card)

    post = []
    if rng.random() < 0.6:
        post.append(rng.choice(STATUS_TRAPS[trap_statuses[0]]).format(subject=subject, speaker=speaker))
    if rng.random() < 0.7:
        post.append(rng.choice(DUE_TRAP_LINES))
    if rng.random() < 0.55:
        post.append(rng.choice(SOFT_TRAPS[trap_statuses[0]]).format(subject=subject))
    if owner != "unassigned" and rng.random() < 0.6:
        post.append(
            rng.choice(
                [
                    f"{decoy_owner} will own the follow-up if the item is reopened later.",
                    f"Follow-up owner: {decoy_owner} (drill only).",
                    f"Assign follow-up to {decoy_owner} only if reopened.",
                ]
            )
        )
    if rng.random() < 0.7:
        post.append(f"After close, someone again named {other_subject} in the corridor.")

    text = " ".join(body + card + post)

    # Provenance against the exact gold strings.
    d_start = text.index(decision)
    decision_span = _span_at(text, decision, d_start)
    if owner == "unassigned":
        owner_span = _anchor(text)
    else:
        # Prefer owner mention inside the gold owner_clause occurrence.
        clause_start = text.index(owner_clause)
        rel = owner_clause.index(owner_surface)
        owner_span = _span_at(text, owner_surface, clause_start + rel)

    raw_id = f"row-{idx:05d}-{family}-{status}"
    hid = hashlib.sha1(f"board-action-v1:{raw_id}".encode()).hexdigest()[:12]
    return {
        "raw_id": raw_id,
        "id": f"ba_{hid}",
        "template_family": family,
        "text": text,
        "action_status": status,
        "owner": owner,
        "due_bucket": due,
        "subject": _normalize_subject(subject),
        "decision_span": decision_span,
        "owner_span": owner_span,
        "speaker": speaker,
        "distractor_status": trap_statuses[0],
    }


def main() -> None:
    rng = random.Random(SEED)
    rows = [build_row(rng, i) for i in range(N_ROWS)]
    df = pd.DataFrame(rows)
    assert df["template_family"].nunique() == len(FAMILIES)
    assert set(df["action_status"]) == set(STATUSES)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows -> {OUT}")
    print(df["action_status"].value_counts().to_dict())
    print(df["due_bucket"].value_counts().to_dict())


if __name__ == "__main__":
    main()
