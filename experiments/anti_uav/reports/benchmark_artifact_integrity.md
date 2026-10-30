# Benchmark artifact integrity: post-shutdown forensic check

Generated 2026-09-23T20:36:59+0530 by an independent, read-only script (JSON parsing and metrics re-implemented from scratch; does not import `anti_uav_repro`). Source data: `benchmark_artifact_integrity.json` (per-sequence detail).

Checks per prediction file: JSON parses; `sequence` field = filename and unique; sequence is in the frozen manifest of the split; `split`/`modality`=IR/`frame_stride`=1; len(res) = len(scores) = len(processed) = `n_frames` = manifest n_frames = label length; `start_frame` = manifest start frame; every box is 4 finite numbers with w,h>0, and every other output is an explicit absent marker; every score finite; `processed` true exactly for frames >= start_frame; `n_processed` consistent. Checks per run: checkpoint SHA-256 on disk now equals the one in `eval_meta.json`; manifest SHA-256 equals the frozen manifest; label files still hash to the manifest; files present form an exact prefix of manifest order (no gaps); ledger entry present for TEST.

Frame indices are positional (list index = frame index), so 'no missing / duplicate frames' reduces to exact length equality with the label file. That holds for every file checked.

## Frozen manifests

| Split | SHA-256 | Sequences | Frames | Label files changed since freeze |
|---|---|---:|---:|---|
| val | `a688bf92c46dfc52…` | 67 | 61999 | 0 |
| test | `1fc351c74f4b29cc…` | 91 | 85374 | 0 |

## Per evaluation

| Evaluation | In authorised scope? | Pred. files | Valid seqs / expected | Frames in valid preds / expected | Schema problems | Ckpt SHA ok | Manifest ok | `timestamp_end` written | metrics.json | Re-score = reported | Classification |
|---|---|---:|---:|---:|---:|---|---|---|---|---|---|
| `normal_ir/val` | YES | 67 | 67 / 67 | 61999 / 61999 | 0 | True | True | True | True | True | **A_COMPLETED_AND_VERIFIED** |
| `dfsc_ir/val` | YES | 67 | 67 / 67 | 61999 / 61999 | 0 | True | True | True | True | True | **A_COMPLETED_AND_VERIFIED** |
| `normal_ir/test` | YES | 91 | 91 / 91 | 85374 / 85374 | 0 | True | True | True | True | True | **A_COMPLETED_AND_VERIFIED** |
| `dfsc_ir/test` | YES | 91 | 91 / 91 | 85374 / 85374 | 0 | True | True | True | True | True | **A_COMPLETED_AND_VERIFIED** |
| `phaseB_normal_ir_seed1/test` | **NO** | 91 | 91 / 91 | 85374 / 85374 | 0 | True | True | True | True | True | **A_COMPLETED_AND_VERIFIED** |
| `phaseB_dfsc_ir_seed1/test` | **NO** | 91 | 91 / 91 | 85374 / 85374 | 0 | True | True | True | True | True | **A_COMPLETED_AND_VERIFIED** |
| `phaseB_dfsc_ir_ism_cross/test (Table VI DFSC-all)` | **NO** | 54 | 54 / 91 | 51009 / 85374 | 0 | True | True | False | False | n/a (no metrics.json) | **C_PARTIAL** |

Maximum absolute difference, independent re-score vs the evaluator's `metrics.json` (mSA, precision, success, all 10 attribute mSAs, every per-sequence SA):

- `normal_ir/val`: 0.0e+00
- `dfsc_ir/val`: 0.0e+00
- `normal_ir/test`: 0.0e+00
- `dfsc_ir/test`: 0.0e+00
- `phaseB_normal_ir_seed1/test`: 0.0e+00
- `phaseB_dfsc_ir_seed1/test`: 0.0e+00

## Checkpoint identity

| Evaluation | Checkpoint | Epoch | SHA-256 (in eval_meta = on disk now) |
|---|---|---:|---|
| `normal_ir/val` | `experiments\anti_uav\runs\normal_ir\weights\last.pt` | 18 | `ab9890d52684730e734449f65ce8cf4747d9a855e18a63ae6e9272101df93180` |
| `dfsc_ir/val` | `experiments\anti_uav\runs\dfsc_ir\weights\last.pt` | 18 | `1e8219c56d4b9bfe75ebb4f716d7174f8b32f924b679e4553ae58054d863b860` |
| `normal_ir/test` | `experiments\anti_uav\runs\normal_ir\weights\last.pt` | 18 | `ab9890d52684730e734449f65ce8cf4747d9a855e18a63ae6e9272101df93180` |
| `dfsc_ir/test` | `experiments\anti_uav\runs\dfsc_ir\weights\last.pt` | 18 | `1e8219c56d4b9bfe75ebb4f716d7174f8b32f924b679e4553ae58054d863b860` |
| `phaseB_normal_ir_seed1/test` | `experiments\anti_uav\runs\phaseB_normal_ir_seed1\weights\last.pt` | 18 | `414ef0164e2bfc6520fb19b2e1d487fbb5df14836433da4923223b24f5032b13` |
| `phaseB_dfsc_ir_seed1/test` | `experiments\anti_uav\runs\phaseB_dfsc_ir_seed1\weights\last.pt` | 18 | `dc0b3e643b53222bc7f24a65a4104b08a4cf748306f082ca535c9269b4899542` |
| `phaseB_dfsc_ir_ism_cross/test (Table VI DFSC-all)` | `experiments\anti_uav\runs\phaseB_dfsc_ir_ism_cross\weights\last.pt` | 18 | `708ce9666ff84f1856b0a9d4fe6d213cc32215c50bb8ee968e908a897840bb5f` |

## The interrupted evaluation

`phaseB_dfsc_ir_ism_cross/test (Table VI DFSC-all)`: 54 of 91 prediction files, all 54 pass every check, and they are an exact prefix of manifest order. Last file written: `20190925_200805_1_8` at 2026-09-23T20:03:46. Every file is written by a single `json.dump` after a sequence finishes tracking, so the 55th sequence (`20190925_200805_1_9`) was being tracked when the machine died and left no file. No truncated file exists. `metrics.json`, `report.md`, `timestamp_end` and the ledger entry are all absent, as expected for a run killed before scoring.

Sequence 53 (`20190925_200805_1_7`) logged 0.1 fps: about 124 minutes of wall time for 958 frames, from 17:58 to 20:02. The machine stalled during that sequence. Its prediction file still passes every structural check. Tracking is causal, so a pause does not change the outputs; the file is structurally sound.

## Comparison with historical (pre-benchmark) evaluations of the same checkpoints

| Evaluation | Historical dir | Same ckpt SHA | Hist. mSA | New mSA | Seqs with any box diff | Frames differing | Present/absent flips | Max box diff (px) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `normal_ir/val` | `experiments/anti_uav/runs/normal_ir/eval_val_IR_last` | True | 79.287413 | 79.287413 | 3/67 | 22 | 0 | 0.030 |
| `dfsc_ir/val` | `experiments/anti_uav/runs/dfsc_ir/eval_val_IR_last` | True | 79.255915 | 79.255898 | 15/67 | 14924 | 0 | 4.537 |
| `normal_ir/test` | `experiments/anti_uav/runs/normal_ir/eval_test_IR_last` | True | 70.579620 | 70.579625 | 91/91 | 27880 | 0 | 1.008 |
| `dfsc_ir/test` | `experiments/anti_uav/runs/dfsc_ir/eval_test_IR_last` | True | 70.286780 | 70.286618 | 91/91 | 85221 | 0 | 137.168 |

The predictions are not bit-identical to the historical ones. The runs used `cudnn.benchmark=True` on GPU, and the historical runs were made at earlier git commits (evaluator code unchanged). mSA agrees to within 0.0002 pt, precision@20 is identical in all four, and success AUC differs by at most 0.00007 pt. The 137 px maximum in dfsc_ir/test is a local trajectory divergence in a few frames that barely moves the score. Treat these as re-measurements that agree within GPU non-determinism, not as new or different results.
