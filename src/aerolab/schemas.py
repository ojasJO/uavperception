"""Versioned JSON schemas for every workspace record. Bump the version string when a field changes meaning.

Unknown values are explicit strings ('unknown', 'not_recorded', 'not_verified') or null -- never invented numbers.
"""
import jsonschema

S = {"type": "string"}
S_NULL = {"type": ["string", "null"]}
INT_NULL = {"type": ["integer", "null"]}

MODEL_STATUS = ["registered", "validated", "experimental", "legacy", "retired", "invalid", "smoke", "external", "unverified"]
INTEGRITY = ["verified_against_record", "hashed_no_independent_record", "mismatch", "missing"]
VERIFY = ["not_verified", "passed", "failed", "not_applicable"]

MODEL_SCHEMA = {
    "$id": "src/aerolab/model/1", "type": "object",
    "required": ["schema", "model_id", "name", "direction", "family_id", "run_id", "checkpoint_tag", "version", "status", "sha256",
                 "size_bytes", "locations", "primary_path", "integrity_status", "task", "adapter", "verification", "evaluations"],
    "properties": {
        "schema": {"const": "src/aerolab/model/1"}, "model_id": S, "name": S, "direction": {"enum": ["aerotrack", "antiuav_repro", "shared"]},
        "family_id": S, "run_id": S, "checkpoint_tag": S, "version": S, "seed": INT_NULL,
        "status": {"enum": MODEL_STATUS}, "aliases": {"type": "array", "items": S}, "role": S_NULL,
        "registrations": {"type": "array", "items": S},
        "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}, "size_bytes": {"type": "integer", "minimum": 0},
        "locations": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["path", "location", "sha256"],
                      "properties": {"path": S, "location": {"enum": ["workspace", "recovery_backup"]}, "sha256": S}}},
        "primary_path": S, "integrity_status": {"enum": INTEGRITY},
        "recorded_hashes": {"type": "array", "items": {"type": "object", "required": ["source", "sha256", "match"]}},
        "task": S, "adapter": S_NULL, "verification": {"type": "object", "required": ["integrity", "load", "inference", "evaluation"],
                                                       "properties": {k: {"enum": VERIFY} for k in ("integrity", "load", "inference", "evaluation")}},
        "evaluations": {"type": "array", "items": S}, "duplicate_group": S_NULL, "parent": S_NULL,
    },
}

FILE_SCHEMA = {
    "$id": "src/aerolab/file/1", "type": "object",
    "required": ["logical_path", "physical_path", "location", "size_bytes", "sha256", "artifact_kind", "family_id", "model_id"],
    "properties": {"logical_path": S, "physical_path": S, "location": {"enum": ["workspace", "recovery_backup"]},
                   "size_bytes": {"type": "integer"}, "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                   "artifact_kind": {"enum": ["model_weights", "export", "detection_cache", "unclassified"]},
                   "family_id": S_NULL, "model_id": S_NULL},
}

DATASET_SCHEMA = {
    "$id": "src/aerolab/dataset/1", "type": "object", "required": ["schema", "id", "name", "roots", "splits", "status"],
    "properties": {"schema": {"const": "src/aerolab/dataset/1"}, "id": S, "name": S, "roots": {"type": "object"}, "splits": {"type": "object"},
                   "status": {"type": "object"}},
}

EXPERIMENT_SCHEMA = {
    "$id": "src/aerolab/experiment/1", "type": "object",
    "required": ["schema", "experiment_id", "direction", "kind", "provenance", "split", "metrics", "test_pass", "source_file"],
    "properties": {
        "schema": {"const": "src/aerolab/experiment/1"}, "experiment_id": S, "direction": {"enum": ["aerotrack", "antiuav_repro", "shared"]},
        "kind": {"enum": ["train", "evaluation", "registered_test_pass", "exploratory_val", "hpc_job", "paper_reported",
                          "released_baseline", "hypothesis_verdict", "studio_evaluation"]},
        "provenance": {"enum": ["historical_import", "paper_reported", "reproduced_from_released_predictions", "studio_run"]},
        "model_ids": {"type": "array", "items": S}, "checkpoint_sha256": S_NULL, "dataset_id": S_NULL, "split": S_NULL,
        "modality": S_NULL, "protocol_id": S_NULL, "metrics": {"type": "object"}, "per_sequence": {"type": ["object", "null"]},
        "seed": INT_NULL, "git_commit": S_NULL, "status": S_NULL, "registration": S_NULL, "test_pass": {"type": "boolean"},
        "source_file": S, "notes": S_NULL, "warnings": {"type": "array", "items": S},
    },
}

# the manifest written by aerolab.evaluation for every studio evaluation (task-brief §9.1 fields)
STUDIO_MANIFEST_SCHEMA = {
    "$id": "src/aerolab/studio_manifest/1", "type": "object",
    "required": ["schema", "experiment_id", "research_direction", "model_id", "checkpoint_path", "checkpoint_sha256", "dataset_id",
                 "dataset_version_or_manifest_hash", "split", "evaluation_protocol_id", "metric_definition_version", "preprocessing_config",
                 "seed", "git_commit", "git_dirty", "environment_reference", "device", "start_time", "completion_time", "status", "metrics",
                 "per_sample_results_path", "logs_path", "registration_reference", "warnings"],
    "properties": {"schema": {"const": "src/aerolab/studio_manifest/1"}, "split": {"enum": ["train", "val"]},
                   "status": {"enum": ["completed", "failed", "partial"]}, "checkpoint_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                   "registration_reference": {"type": ["string", "null"]}, "warnings": {"type": "array", "items": S}},
}

PREDICTION_SCHEMA = {
    "$id": "src/aerolab/prediction/1", "type": "object",
    "required": ["frame", "present", "box_xyxy", "score", "model_id", "checkpoint_sha256"],
    "properties": {"frame": {"type": "integer", "minimum": 0}, "present": {"type": ["boolean", "null"]},
                   "box_xyxy": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                   "score": {"type": ["number", "null"]}, "mode": S_NULL, "track_id": INT_NULL, "inference_ms": {"type": ["number", "null"]},
                   "model_id": S, "checkpoint_sha256": S, "warnings": {"type": "array", "items": S}},
}

SCHEMAS = {s["$id"]: s for s in (MODEL_SCHEMA, FILE_SCHEMA, DATASET_SCHEMA, EXPERIMENT_SCHEMA, STUDIO_MANIFEST_SCHEMA, PREDICTION_SCHEMA)}


def validate(record: dict, schema: dict):
    """raises jsonschema.ValidationError with the offending path"""
    jsonschema.validate(record, schema, cls=jsonschema.Draft202012Validator)


def errors(record: dict, schema: dict) -> list[str]:
    v = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in v.iter_errors(record)]
