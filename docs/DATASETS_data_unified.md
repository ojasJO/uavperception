# Dataset card — corpus `data_unified`

*Generated 2026-10-05T02:28:47+0530 by `tools/dataset_card.py` from `b_pipeline\data_unified\manifest.csv` and its label cache; never edit by hand.*

Class mapping: {'0': "uav_drone (Anti-UAV, CST, Det-Fly 'Drone')", '1': "bird_hard_negative (Det-Fly 'Bird')"}. Preprocessing: YOLO-normalised labels; frames read from the source extraction (Anti-UAV: JPEG decodes of the MP4s); training letterboxes to 640.

| dataset / modality / split | seq | frames | with target | drone boxes | bird boxes | res | box diag median (p05–p95) px | tiny / small / normal / large | micro area | motion px/frame median (p95) | invisible runs (mean / max len) |
|---|---:|---:|---:|---:|---:|---|---|---|---:|---|---|
| anti_uav/IR/test | 91 | 85,374 | 84,178 | 84,178 | 0 | 640×512 | 55.6 (33.8–94.6) | 0.0 % / 3.3 % / 28.0 % / 68.7 % | 0.0 % | 4.03 (22.0) | 57 (21.0 / 217) |
| anti_uav/IR/train | 160 | 149,528 | 148,189 | 148,189 | 0 | 640×512 | 58.6 (34.0–99.3) | 0.0 % / 1.7 % / 32.9 % / 65.4 % | 0.0 % | 3.64 (16.5) | 51 (26.3 / 229) |
| anti_uav/IR/val | 67 | 61,999 | 60,548 | 60,548 | 0 | 640×512 | 57.4 (33.3–106.9) | 0.0 % / 2.0 % / 26.1 % / 71.9 % | 0.0 % | 3.16 (16.5) | 23 (63.1 / 901) |
| anti_uav/RGB/test | 91 | 85,374 | 79,699 | 79,699 | 0 | 1920×1080 | 144.1 (58.0–247.6) | 0.0 % / 0.1 % / 1.8 % / 98.1 % | 0.2 % | 9.12 (57.1) | 75 (75.7 / 770) |
| anti_uav/RGB/train | 160 | 149,528 | 142,106 | 142,106 | 0 | 1920×1080 | 124.7 (64.8–253.7) | 0.0 % / 0.1 % / 1.9 % / 98.0 % | 0.5 % | 9.01 (41.5) | 105 (70.7 / 564) |
| anti_uav/RGB/val | 67 | 61,999 | 58,262 | 58,262 | 0 | 1920×1080 | 135.2 (65.5–242.2) | 0.0 % / 0.0 % / 0.4 % / 99.5 % | 0.1 % | 8.02 (40.1) | 32 (116.8 / 903) |
| cst/IR/test | 60 | 72,175 | 59,250 | 59,250 | 0 | 640×512 | 14.3 (7.8–30.1) | 17.5 % / 77.3 % / 4.4 % / 0.8 % | 57.4 % | 0.76 (10.0) | 316 (40.9 / 513) |
| cst/IR/train | 84 | 90,012 | 75,474 | 75,474 | 0 | 640×512 | 14.7 (6.1–26.4) | 27.8 % / 68.8 % / 3.3 % / 0.0 % | 52.8 % | 0.55 (4.7) | 178 (81.7 / 841) |
| det_fly/RGB/test | 1,567 | 1,567 | 1,567 | 1,470 | 1,627 | 640×640 | 33.8 (11.3–105.8) | 4.6 % / 37.4 % / 28.5 % / 29.4 % | 11.3 % | — | — |
| det_fly/RGB/train | 4,698 | 4,698 | 4,698 | 4,407 | 5,006 | 640×640 | 33.1 (12.2–108.7) | 3.4 % / 40.5 % / 26.6 % / 29.4 % | 11.2 % | — | — |
| det_fly/RGB/val | 1,565 | 1,565 | 1,565 | 1,468 | 1,991 | 640×640 | 29.8 (12.0–100.5) | 3.5 % / 47.3 % / 25.9 % / 23.3 % | 12.4 % | — | — |

## Sources, terms, known defects

- **anti_uav** — archive `datasets/1_anti-uav/Anti-UAV-RGBT.zip`; source https://github.com/ZhaoJ9014/Anti-UAV (Jiang et al., arXiv:2101.08466); terms: academic research use per the Anti-UAV benchmark; no licence file inside the archive (verify before redistribution). IR 640x512 + RGB 1920x1080 per clip; official 160/67/91 sequences per modality; released box count ~2 % below the paper's Table I; VAL shares all 26 recording sessions with TRAIN
- **cst** — archive `datasets/2_cst-antiuav (z01-z07) + datasets/2_cst-antiuav_part2 (z08-z10, .zip)`; source https://github.com/PCwenyue/CST-Anti-UAV (Xie et al., ICCVW 2025, arXiv:2507.23473); terms: academic and non-commercial research (dataset README / DATASETS_TERMS). corpus v1 was built from parts z01-z07 only (84/120 train, 0/40 val, 60/60 test sequences); corpus v2 from all 11 parts (CRC-verified). Official release 120/40/60.
- **det_fly** — archive `datasets/3_det-fly/Det-Fly.v5i.yolov11.zip`; source Roboflow export of https://github.com/Jake-WU/Det-Fly (Zheng et al., IEEE RA-L 2021); terms: CC BY 4.0 (Roboflow export), research use only per the export disclaimer; cite the original. random image-level export split (VAL frames adjacent to TRAIN frames: 1,224 of 1,469); images stretched to 640x640; the only source of birds
