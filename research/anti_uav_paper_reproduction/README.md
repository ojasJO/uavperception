# Direction B — reproduction of *Anti-UAV: A Large Multi-Modal Benchmark for UAV Tracking* (DFSC on GlobalTrack)

**Paper.** Jiang et al., arXiv:2101.08466v3 (2021; journal version IEEE TMM). Local copy: `anti_uav_research_paper.pdf`.
Official code: github.com/ZhaoJ9014/Anti-UAV. Base tracker: GlobalTrack (github.com/huanglianghua/GlobalTrack).

**Question.** Does a faithful re-implementation of the paper's DFSC training on GlobalTrack reproduce its Protocol II infrared numbers,
and does DFSC beat plain fine-tuning?

**Not to be confused with** the CST Anti-UAV benchmark (arXiv:2507.23473). That paper is a benchmark paper whose protocol scores
Direction A's AeroTrack models. It is not reproduced as a method here, because it proposes none.

Code: `experiments/anti_uav/` (self-contained: `src/anti_uav_repro`, configs, frozen manifests, append-only TEST ledger, the original
VAL demo `experiments/anti_uav/demo/`). The studio's adapter `globaltrack_repro` calls exactly this tracker.

## What is faithful, inherited, or substituted

`docs/ANTI_UAV_REPRODUCTION_SPEC.md` tags every setting as paper-stated, inherited from GlobalTrack, or substituted (S1-S6). Among the
substitutions: a torchvision re-implementation of QG-RCNN, initialised from torchvision COCO Faster R-CNN (`model.init` in every run
config), 1 GPU × batch 2 with a linearly scaled learning rate, and the final-epoch checkpoint with no selection.

| layer | what it is | label |
|---|---|---|
| paper's reported numbers | transcribed from Table V (`experiments/anti_uav/reports/paper_reported_results.json`) | **paper** (never ours) |
| `normal_ir`, `dfsc_ir` (seed 0) | Phase A faithful reproduction, Protocol II IR | **reproduction** |
| `phaseB_*_seed1` | seed replication; TEST evaluated outside the then-authorised scope (supplementary) | **replication** |
| `phaseB_*_lr0.02_accum8` (H3), H4 native scale, absence threshold | follow-up hypotheses | **exploratory** |
| `phaseB_dfsc_ir_ism_cross` | DFSC-all ablation | **invalid: no valid result** |
| `normal_rgb`, `dfsc_rgb`, `largescale_globaltrack` | A2 visible modality and the paper's "large-scale" (zero-shot GlobalTrack) row | **reproduction (A2)** |

## Results (generated — do not edit by hand)

<!-- BEGIN GENERATED: tools/render_workspace_docs.py -->
| method | split | paper mSA (IR) | ours mSA (IR) | ours: record |
|---|---|---:|---:|---|
| normal (seed 0) | val | 79.60 | 79.29 | `ledger:eval:anti_uav/repro/normal_ir/seed0/val` |
| normal (seed 0) | test | 65.36 | 70.58 | `ledger:eval:anti_uav/repro/normal_ir/seed0/test` |
| DFSC (seed 0) | val | 80.09 | 79.26 | `ledger:eval:anti_uav/repro/dfsc_ir/seed0/val` |
| DFSC (seed 0) | test | 66.04 | 70.29 | `ledger:eval:anti_uav/repro/dfsc_ir/seed0/test` |
| normal (seed 1, replication) | val | — | 79.36 | `ledger:eval:anti_uav/repl/phaseB_normal_ir_seed1/val` |
| normal (seed 1, replication) | test | — | 70.22 | `ledger:eval:anti_uav/repl/phaseB_normal_ir_seed1/test` |
| DFSC (seed 1, replication) | val | — | 79.43 | `ledger:eval:anti_uav/repl/phaseB_dfsc_ir_seed1/val` |
| DFSC (seed 1, replication) | test | — | 70.14 | `ledger:eval:anti_uav/repl/phaseB_dfsc_ir_seed1/test` |

Paper numbers are transcribed from Table V (`experiments/anti_uav/reports/paper_reported_results.json`), labelled `paper_reported` in the registry and never ranked against ours in the studio (no per-sequence data). Authoritative report: `experiments/anti_uav/reports/FINAL_PROTOCOL_II_IR_BENCHMARK_REPORT.md`.
<!-- END GENERATED -->

## Status and limitations

- **VAL reproduced** within the pre-registered 1-point band. **TEST not reproduced: it scores above the paper.** Absence handling and the
  metric implementation were ruled out as causes; the label version is plausible but untestable; H3 / H4 are explored in Phase B.
- **DFSC effect not observed** across two seeds.
- The tracker never reports absence, so every target-absent frame scores 0 under Eq. 1.
- TEST is spent for the authorised runs (`experiments/anti_uav/reports/test_evaluation_ledger.jsonl`). The studio runs VAL only.

## Next justified steps

Remaining untested explanations of the TEST surplus (`docs/ANTI_UAV_COMPARISON.md` §Discrepancy) need their own VAL-first registration.
No further TEST use without one.
