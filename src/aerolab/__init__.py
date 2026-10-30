"""aerolab — shared workspace layer for the two AeroTrack research directions.

Light by design: importing aerolab (or aerolab.registry) never imports torch, ultralytics or the experiment code. Heavy
dependencies are imported inside adapters only when a model is actually loaded.

    aerolab.paths      repository paths (one place)
    aerolab.hashing    SHA-256 with a (path, size, mtime) cache
    aerolab.schemas    versioned JSON schemas of every registry / record type + validation
    aerolab.registry   read-only access to results/registry/*.json (models, files, datasets, experiments, protocols, papers)
    aerolab.test_lock  the TEST-split lock (consumed registrations; the studio never runs a TEST pass)
    aerolab.compare    protocol compatibility of two result records
    aerolab.adapters   model adapters wrapping the existing pipelines
    aerolab.evaluation small VAL evaluations with complete manifests (results/runs/studio/)
    aerolab.health     read-only artifact / environment checks
"""
__version__ = "1.0.0"
