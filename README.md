# Board Action Card (`board_action_slots`)

Synthetic meeting-minute **action card** dataset for an Eris / Shipd NLP challenge.

- **License:** [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) (public domain dedication)
- **Generator:** `generate_raw.py` (deterministic, seed `20261009`)
- **Task:** decision status + owner + due bucket + subject + provenance spans under discussion distractors

```bash
python3 generate_raw.py   # writes raw/minutes.csv
```

See `dataset_description.md` for provenance, limitations, and how this differs from DocIE / M&A deal-card extraction.
