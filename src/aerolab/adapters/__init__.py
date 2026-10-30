"""Adapter registry: adapter id (results/registry/catalog/families.json 'adapter') -> class. A family with adapter null has
recorded results only (browsable, not runnable)."""
from .base import AdapterError, Compatibility, ModelAdapter

ADAPTERS = {
    "aerotrack_detector": "aerolab.adapters.aerotrack_detector:AeroTrackDetectorAdapter",
    "globaltrack_repro": "aerolab.adapters.globaltrack:GlobalTrackAdapter",
    "trackc_forecaster": "aerolab.adapters.forecaster:TrackCForecasterAdapter",
}


def get_adapter(adapter_id: str, registry=None) -> ModelAdapter:
    if adapter_id not in ADAPTERS:
        raise AdapterError(f"no adapter {adapter_id!r}: this model has recorded results only")
    import importlib
    mod, cls = ADAPTERS[adapter_id].split(":")
    return getattr(importlib.import_module(mod), cls)(registry=registry)


def compatibility(model_record: dict, dataset_id: str, modality: str | None = None, registry=None) -> Compatibility:
    a = model_record.get("adapter")
    if not a:
        return Compatibility(False, "recorded results only: no inference adapter for this family (see its recorded experiments)")
    return get_adapter(a, registry).can_run(model_record, dataset_id, modality)


__all__ = ["ADAPTERS", "AdapterError", "Compatibility", "ModelAdapter", "compatibility", "get_adapter"]
