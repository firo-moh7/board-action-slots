# Dataset Description

## Overview

This dataset supports a **board action card** slot-filling task. Each row is one short synthetic meeting-minute paragraph. Models must fill decision status, owner, due bucket, subject, and character-offset provenance for the decision sentence and owner mention.

This is decision-status binding under discussion distractors. It is not DocIE/ACE event extraction, not open NER, and not M&A press-release economics slot filling.

## Where the rows come from

All text is **synthetic** (template-generated English minutes) produced by `generate_raw.py` in this package.

- No scraped corporate minutes
- No PII from real persons (names are randomly composed from common given/family pools)
- License: **CC0-1.0** (dedicated to the public domain)

Competition-specific columns and the template-family holdout split are constructed in `prepare.py`.

## How this differs from related public work

- **DocIE / ACE / ERE**: open or large event schemas with trigger/argument annotation. This challenge grades a **fixed six-slot action card** with adversarial discussion plants, not multi-event IE F1.
- **M&A deal-card / BUSTER-style party extraction**: buyer/seller/acquired + economics. This challenge has no parties, consideration, or USD value; the hard part is binding status to the **decision** sentence when discussion uses conflicting status cues.
- **Meeting summarization / action-item extraction benchmarks** (e.g. AMI-style summaries): usually free-form or multi-item lists. Here the output schema is closed, jointly graded, and deliberately single-card.
- **NLI / claim support** (MULTIATTR, lemma-style peer drafts): no paired drafts or support_attr labels; one minute → one card.

## Preparation transforms

- Generates or loads `raw/minutes.csv` with gold slots + `template_family`
- Deduplicates normalized text
- Whole-family holdout: ~1/3 of rhetoric families → test, remainder → train
- Opaque hashed ids (`ba_` + hex)
- Public/private visibility assigned per test family

Counts after preparation (seed 20261009): about **800 train / 400 test** (exact counts depend on family sizes after dedupe); see `private/split_meta.csv`.

## File structure after prepare

- `public/train.csv` — labeled training rows
- `public/test.csv` — `id`, `text` only
- `public/sample_submission.csv` — placeholders
- `private/answers.csv` — gold labels + `visibility`
- `private/split_meta.csv` — family membership summary

## Required submission columns

`id`, `action_status`, `owner`, `due_bucket`, `subject`, `decision_span`, `owner_span`

## Known limitations

- Domain is short English synthetic minutes; transfer to real corporate minutes is not guaranteed
- Stamp tokens are desk/family-local; decoy mark-key tables are planted alongside a binding key
- Twin surviving stamps share mark text; only the card-region occurrence is the graded `decision_span` (exact offsets)
- Policy-override lines can change the graded status; draft/superseded override wording must be ignored
- ~18% of rows have residual status-annotator disagreement with no extra textual cue (soft ceiling)
- Owner spans for `unassigned` use a document-start anchor convention
- Random name pools may collide across rows; ids remain the only join key

## Source attribution

Synthetic data generated offline for this challenge. Generator scripts are **not** part of the agent-facing package.

Source repository: https://github.com/firo-moh7/board-action-slots
