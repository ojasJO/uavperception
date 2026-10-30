**What improved.**
- [measured] Once every configuration starts from the same pretrained function, all three candidates are slightly ahead of
  `stock` in every fold (9 of 9 fold comparisons positive), i.e. the large penalties the repair grid attributed to SPD-Conv and
  to the temporal stack were produced by the initialisation, not by the architecture.
- [rule] Only the motion-matched temporal stack (`temporal_fp_g4`: t−4, t, t+4 frames, COCO stem on the centre frame) passes the
  pre-registered rule: CST drone AP50 +0.045 on held-out folds (+0.045 / +0.036 / +0.054; bootstrap 95 % CI of the mean
  [+0.021, +0.072]), Anti-UAV unchanged (−0.002). It is also ahead on the official VAL in every fold (CST +0.070 to +0.094) and,
  on its single test evaluation, above stock YOLO11n's single test evaluation (CST 0.521 vs 0.456, pooled drone 0.718 vs 0.684).

**What did not improve.**
- [rule] SPD-Conv with a fair start (`spd_fp`): +0.014, CI [−0.005, +0.034] — below the margin; not supported. With a 55 %
  larger network and no reproducible gain, it is not justified.
- [rule] The original t−1, t, t+1 stack with a fair start (`temporal_fp_g1`): +0.009, CI [−0.005, +0.023] — not supported.
- [fact] The training-budget question (40 vs 20 epochs) was not run: the pre-registered 24 h guard skipped it.

**Why the gap matters (inference).** CST cameras are static and its drones move about 0.5 px per frame (D1). At g = 1 the three
frames are nearly identical, so there is no motion to learn; at g = 4 the displacement is about 2 px, enough for the network to
use the motion of small, low-contrast thermal targets against a still background. Anti-UAV targets are large and visible in a
single frame, which matches the unchanged Anti-UAV score. [hypothesis] The gain comes from motion contrast, not from extra
capacity: `temporal_fp_g1` has the same network and input size and gains almost nothing.

**Credibility.** [inference] Moderate-to-good for the development claim: one change, chosen from a measurement (not a search),
pre-registered, paired against the reference with the same seed and the same starting function, replicated on three
session/sequence-disjoint folds, and the CI excludes zero. The test result agrees in direction. It is **not** a paired test
comparison (stock's test predictions were not kept, and re-scoring stock on test would break the once-only rule): stock's CST
test value (0.456) lies just inside the selected model's own CST CI [0.442, 0.597], so the test alone would not prove the gain.

**Limitations.**
1. One seed per fold (seed = fold index); fold-to-fold spread mixes data and optimisation noise (sd of Δ 0.009).
2. The CST development folds contain only the four training scene categories. On the test split the selected model scores 0.617
   on seen categories but 0.372 on the two unseen ones (`urban-areas` 0.242) — whether the temporal gain holds on new scene types
   is untested, and new scene types are the dominant remaining failure.
3. Deployment cost: g = 4 needs 4 future frames (+160 ms at 25 fps, vs +40 ms for g = 1 and 0 for a single frame); training reads
   three distinct frames per sample (≈ 1.35× the stock training time).
4. All models were still improving at epoch 20 (monitoring mAP50 slopes ≈ +0.005 to +0.010 per epoch); rankings at a longer
   budget are untested.
5. Det-Fly offers little independent evidence (two flights; its "temporal" input is always t,t,t).
6. Only g ∈ {0, 1, 4} was tested; the optimum gap is unknown and was deliberately not searched.

**Recommended next step.** Adopt stock YOLO11n with a centre-frame-initialised t−4, t, t+4 input as the AeroTrack detector; drop
SPD-Conv. Before claiming more, test — on the same folds, pre-registered — (a) a causal variant (t−8, t−4, t) that removes the
look-ahead latency, (b) three seeds per fold at 20 and 40 epochs, and (c) a leave-category-out CST protocol to measure unseen
scene types. Do not re-evaluate the test split for any of these; a new final candidate would need its own single test pass.

**Final conclusion.** [conclusion] The investigation found an implementation/training issue (initialisation) that invalidated
the repair grid's architectural verdicts, fixed it, and under a controlled protocol found one change with credible evidence of
better generalisation to unseen sequences — a temporal gap matched to the measured motion (+0.045 CST drone AP50, CI
[+0.021, +0.072]) — while SPD-Conv and the original t±1 stack remain unsupported.
