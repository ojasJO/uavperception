# Tracker comparison — aerotrack_spd_finetuned — VAL (5 longest sequences per dataset (track_eval.pick_sequences), first 1200 frames)

Generated 2026-09-21T16:34:33+0530; weights sha256 `72de9bba9fec`; detections at conf 0.05; 10 sequences, 10578 frames.

| tracker | MOTA | IDF1 | ID switches | ID sw @ occlusion re-entry | FP | FN | predicted IDs (GT: 1/seq) |
|---|---|---|---|---|---|---|---|
| bytetrack | 0.5957 | 0.3086 | 578 | 0 | 1326 | 1898 | 279 |
| sort | 0.6446 | 0.3066 | 375 | 0 | 368 | 2599 | 258 |
| argmax@0.5 | 0.7349 | 0.8152 | 0 | 0 | 254 | 2239 | 10 |
| argmax@0.25 | 0.7832 | 0.8624 | 0 | 0 | 350 | 1689 | 10 |
| argmax@0.1 | 0.7783 | 0.8691 | 0 | 0 | 601 | 1484 | 10 |
| greedy1@0.25 | 0.7766 | 0.8592 | 0 | 0 | 381 | 1720 | 10 |
| greedy1@0.1 | 0.7574 | 0.8585 | 0 | 0 | 699 | 1582 | 10 |

## Per dataset (MOTA / IDF1)

| dataset | bytetrack | sort | argmax@0.5 | argmax@0.25 | argmax@0.1 | greedy1@0.25 | greedy1@0.1 |
|---|---|---|---|---|---|---|---|
| anti_uav | 0.635 / 0.167 | 0.805 / 0.184 | 0.945 / 0.972 | 0.955 / 0.977 | 0.955 / 0.977 | 0.942 / 0.971 | 0.926 / 0.963 |
| cst | 0.551 / 0.450 | 0.463 / 0.430 | 0.497 / 0.659 | 0.588 / 0.747 | 0.578 / 0.761 | 0.588 / 0.747 | 0.566 / 0.754 |

**Rule:** ByteTrack is kept only if it beats argmax@0.25 on MOTA and IDF1 on every dataset.  **Verdict:** ByteTrack RETIRED (a no-association argmax baseline is at least as good)
