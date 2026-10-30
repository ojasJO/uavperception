# Anti-UAV dataset verification

Generated 2026-09-19T19:48:31 (git 1debecdf0c512c0e641afe9927136f8f2064c016, dirty=True). Verdict: **OK**

- data root: `A:\AEROTRACK_OMEN\data\anti_uav`
- frames root: `A:\AEROTRACK_OMEN\b_pipeline\data_unified\anti_uav\images`

## Counts

| split | modality | sequences | frames | exist (boxes) | absent | seq len | resolution | start>0 | filter pass | zero-trainable seqs |
|---|---|---|---|---|---|---|---|---|---|---|
| train | IR | 160 | 149528 | 148368 | 1160 | 28–1000 | {'640x512': 160} | 1 | 0.998 | 0 |
| train | RGB | 160 | 149528 | 142192 | 7336 | 28–1000 | {'1920x1080': 160} | 7 | 0.998 | 0 |
| val | IR | 67 | 61999 | 60620 | 1379 | 39–1000 | {'640x512': 67} | 4 | 0.998 | 0 |
| val | RGB | 67 | 61999 | 58302 | 3697 | 39–1000 | {'1920x1080': 67} | 6 | 0.999 | 0 |
| test | IR | 91 | 85374 | 84221 | 1153 | 139–1000 | {'640x512': 91} | 3 | 0.999 | 0 |
| test | RGB | 91 | 85374 | 79724 | 5650 | 139–1000 | {'1920x1080': 91} | 6 | 0.998 | 0 |

Paper (Table I): 318 pairs, 585.9k boxes; train 160 seqs / 294.4k boxes; test 91 / 168.4k. Repo totals above are ~2% lower (documented discrepancy).

## Object size (exist frames)

| split | modality | boxes | √(wh) mean px | √(wh) p5/50/95 px | w p5/50/95 | h p5/50/95 | rel. area % p5/50/95 | LR (<400 px²) | side<10 px |
|---|---|---|---|---|---|---|---|---|---|
| train | IR | 148368 | 38.9 | 23.2/36.9/59.3 | 25.0/51.0/91.0 | 19.0/28.0/48.0 | 0.165/0.416/1.073 | 1.9% | 0.2% |
| train | RGB | 142192 | 85.0 | 44.3/79.9/136.5 | 51.0/109.0/239.0 | 33.0/58.0/92.0 | 0.095/0.308/0.898 | 0.1% | 0.1% |
| val | IR | 60620 | 39.4 | 22.8/37.3/64.0 | 25.0/49.0/94.0 | 18.0/29.0/47.0 | 0.158/0.425/1.249 | 2.0% | 0.2% |
| val | RGB | 58302 | 87.9 | 45.2/85.5/143.6 | 51.0/115.0/230.0 | 34.0/61.0/102.0 | 0.098/0.352/0.994 | 0.1% | 0.1% |
| test | IR | 84221 | 38.4 | 22.9/37.1/59.0 | 26.0/45.0/87.0 | 18.0/29.0/49.0 | 0.160/0.420/1.062 | 3.5% | 0.1% |
| test | RGB | 79724 | 91.0 | 40.1/93.0/146.4 | 39.0/122.0/232.0 | 37.0/69.0/119.0 | 0.078/0.417/1.034 | 0.1% | 0.0% |

## Attributes (sequences per attribute)

- train (IR labels): {'FM': 68, 'LI': 70, 'LR': 23, 'OC': 5, 'OV': 17, 'SV': 34, 'TC': 97, 'TC-EASY': 57, 'TC-HARD': 7, 'TC-MID': 33}
- val (IR labels): {'FM': 23, 'LI': 29, 'LR': 10, 'OC': 5, 'OV': 4, 'SV': 17, 'TC': 32, 'TC-EASY': 17, 'TC-HARD': 7, 'TC-MID': 8}
- test (IR labels): {'FM': 31, 'LI': 18, 'LR': 15, 'OC': 2, 'OV': 18, 'SV': 18, 'TC': 63, 'TC-EASY': 14, 'TC-HARD': 18, 'TC-MID': 31}

## Start frames > 0 (tracker init frame = first exist frame + official corrections)

- train/IR: [('20190925_130434_1_3', 11)]
- train/RGB: [('20190925_130434_1_3', 32), ('20190925_222534_1_6', 402), ('20190926_103046_1_6', 122), ('20190926_133516_1_8', 5), ('20190926_193515_1_5', 343), ('20190926_193515_1_6', 56), ('20190926_195921_1_4', 64)]
- val/IR: [('20190925_130434_1_2', 16), ('20190925_194211_1_3', 61), ('20190925_213001_1_1', 148), ('20190926_183941_1_5', 11)]
- val/RGB: [('20190925_130434_1_2', 9), ('20190925_194211_1_3', 150), ('20190925_194211_1_4', 1), ('20190925_205804_1_5', 23), ('20190925_222534_1_5', 109), ('20190926_183941_1_5', 14)]
- test/IR: [('20190925_111757_1_8', 52), ('20190925_193610_1_1', 7), ('20190925_200805_1_8', 3)]
- test/RGB: [('20190925_111757_1_8', 53), ('20190925_134301_1_6', 434), ('20190925_134301_1_7', 770), ('20190926_102042_1_3', 89), ('20190926_102042_1_6', 20), ('20190926_134054_1_1', 150)]

## Cross-split integrity

- sequence-name overlap: {'train&val': 0, 'train&test': 0, 'val&test': 0}
- byte-identical label files across sequences: 0
- byte-identical first frames across sequences: 0
- source-video ids per split: {'train': 27, 'val': 26, 'test': 10}; overlap: {'train&val': 26, 'train&test': 0, 'val&test': 0}
- The paper states train and val come from non-overlapping clips of the SAME videos (expected video-id overlap), while test is from independent videos (video-id overlap with train/val must be 0).

## Manifests

- `antiuav_IR_train.manifest.json`: verified sha256 `26c7ea09143c3c64…`
- `antiuav_RGB_train.manifest.json`: verified sha256 `e591591412b533cd…`
- `antiuav_IR_val.manifest.json`: verified sha256 `a688bf92c46dfc52…`
- `antiuav_RGB_val.manifest.json`: verified sha256 `0a9a26d4e99505b1…`
- `antiuav_IR_test.manifest.json`: verified sha256 `1fc351c74f4b29cc…`
- `antiuav_RGB_test.manifest.json`: verified sha256 `2ccc2b0c1b60fc5f…`

## Problems

- none

Elapsed 9.1 s.
