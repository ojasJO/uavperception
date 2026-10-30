# X1 — the AeroTrack stack under the CST benchmark's protocol (CST test, single pass)

*Generated 2026-10-12T05:38:57+0530 by `scripts/x1_final.py` under `experiments/aerotrack_v3/preregistration_X1_cst_protocol_matched.json`. Candidate `v3_n__causal_g4_cst_s0_e20` (CST-only training, window from the v2 decision), tracker settings frozen on CST VAL.*

- **X1-a:** STE mSA **60.08** [56.05, 63.99] vs published best 35.92 (GlobalTrack; SiamDT 35.84; MixFormer-V2 29.39) -> **exceeds the best published CST tracker**
- **X1-b:** tiny-target frame SA 39.88 [30.20, 49.02] over 10,354 frames (published: GlobalTrack 15.83, best PrDiMP 28.20)
- argmax mSA 59.70; CST drone AP50 0.6193067507949498
- X1-c: {'v2_candidate': 'v2_causal_g4_s0_e20', 'v2_cst_ste_mSA': 56.554081545589575, 'x1_cst_ste_mSA': 60.081252017353535, 'note': 'descriptive, unpaired across passes'}
