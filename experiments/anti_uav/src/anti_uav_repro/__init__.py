"""anti_uav_repro — faithful reproduction of DFSC / GlobalTrack on the Anti-UAV benchmark.

Layout
------
data/       sequence index, split guard, manifests, pair sampling, transforms
models/     QG-RCNN (torchvision Faster R-CNN + GlobalTrack modulators) and DFSC losses
engine/     training loop and sequence-level tracker inference
eval/       state-accuracy / precision / success metrics and reports
utils/      seeding, environment + git snapshots, hashing

The reproduction spec is docs/ANTI_UAV_REPRODUCTION_SPEC.md; nothing here is allowed to
contradict it silently.
"""

__version__ = "0.1.0"
