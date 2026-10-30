# Early AeroTrack-Net training launch records (preserved 2026-09-24)

`c_model/train.py` writes background-launch state to `runs/_bg/`, which is git-ignored scratch. That folder was deleted
during repository cleanup on 2026-09-24. Two of its files were copied here first:

| File | Why it was kept |
|---|---|
| `train_tuning.json` | Augmentation and sampler weights used by the 2026-09-04 background training launch. This is **not** recorded in any run's `args.json`. |
| `train_results.json` | Summary of the 2026-09-06 fine-tune stage: `runs/aerotrack_spd_finetuned`, best epoch 23, fitness 0.6683. The same values are in that run's `results.csv`; this copy is for convenience. |

Paths inside these files are absolute paths from the original machine.
