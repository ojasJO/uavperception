"""Read-only access to the generated registry (registry/*.json, *.jsonl) and the curated catalog.

    from aerolab.registry import Registry
    R = Registry()                       # loads once; cheap (no torch)
    R.model("aerotrack.detector.cst_x:v3_n__causal_g4_cst_s0_e20@last")
    R.versions("aerotrack.detector.cst_x")
    R.latest("aerotrack.detector.v2"), R.best_validated("aerotrack.detector.cst_x", "sot.mSA.cst_official.v1", "mSA", split="test")
"""
import json
from functools import cached_property
from pathlib import Path

from .paths import CATALOG, REGISTRY, REPO, STUDIO_INDEX


def _load(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _jsonl(p: Path):
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


class RegistryMissing(FileNotFoundError):
    pass


class Registry:
    def __init__(self, root: Path = REGISTRY):
        self.root = root
        if not (root / "models.json").exists():
            raise RegistryMissing("registry/models.json not found -- run: venv\\Scripts\\python.exe tools\\build_workspace_registry.py")

    # ---------------------------------------------------------------- raw tables
    @cached_property
    def models_doc(self):
        return _load(self.root / "models.json", {})

    @cached_property
    def models(self) -> list[dict]:
        return self.models_doc.get("models", [])

    @cached_property
    def by_id(self) -> dict:
        return {m["model_id"]: m for m in self.models}

    @cached_property
    def historical_experiments(self) -> list[dict]:
        return _jsonl(self.root / "experiments.jsonl")

    def studio_experiments(self) -> list[dict]:
        """re-read on every call: new studio evaluations appear without a rebuild"""
        return _jsonl(STUDIO_INDEX)

    def experiments(self) -> list[dict]:
        return self.historical_experiments + self.studio_experiments()

    @cached_property
    def datasets(self) -> list[dict]:
        return (_load(self.root / "datasets.json", {}) or {}).get("datasets", [])

    @cached_property
    def protocols(self) -> dict:
        return {p["id"]: p for p in (_load(CATALOG / "protocols.json", {}) or {}).get("protocols", [])}

    @cached_property
    def papers(self) -> dict:
        return {p["id"]: p for p in (_load(CATALOG / "papers.json", {}) or {}).get("papers", [])}

    @cached_property
    def directions(self) -> dict:
        return {d["id"]: d for d in (_load(CATALOG / "directions.json", {}) or {}).get("directions", [])}

    @cached_property
    def families(self) -> dict:
        return {f["id"]: f for f in (_load(CATALOG / "families.json", {}) or {}).get("families", [])}

    @cached_property
    def findings(self) -> dict:
        return _load(CATALOG / "findings.json", {}) or {}

    @cached_property
    def lineage(self) -> dict:
        return _load(self.root / "model_lineage.json", {}) or {}

    @cached_property
    def duplicates(self) -> list[dict]:
        return (_load(self.root / "duplicates.json", {}) or {}).get("groups", [])

    @cached_property
    def summary(self) -> dict:
        return _load(self.root / "build_summary.json", {}) or {}

    # ---------------------------------------------------------------- queries
    def model(self, model_id: str) -> dict:
        try:
            return self.by_id[model_id]
        except KeyError:
            raise KeyError(f"unknown model id {model_id!r}") from None

    def dataset(self, dataset_id: str) -> dict:
        for d in self.datasets:
            if d["id"] == dataset_id:
                return d
        raise KeyError(f"unknown dataset id {dataset_id!r}")

    def experiment(self, experiment_id: str) -> dict:
        for e in self.experiments():
            if e["experiment_id"] == experiment_id:
                return e
        raise KeyError(experiment_id)

    def versions(self, family_id: str) -> list[dict]:
        return sorted((m for m in self.models if m["family_id"] == family_id), key=lambda m: (m["run_id"], m["checkpoint_tag"]))

    def experiments_for(self, model_id: str) -> list[dict]:
        return [e for e in self.experiments() if model_id in (e.get("model_ids") or [])]

    def latest(self, family_id: str) -> dict | None:
        """most recently modified workspace checkpoint of the family (file mtime; 'latest' says nothing about quality)"""
        best, t = None, -1.0
        for m in self.versions(family_id):
            p = REPO / m["primary_path"]
            if m["status"] in ("smoke", "invalid") or not p.exists():
                continue
            mt = p.stat().st_mtime
            if mt > t:
                best, t = m, mt
        return best

    def best_validated(self, family_id: str, protocol_id: str, metric: str, split: str = "val", **match) -> dict | None:
        """best recorded value of `metric` under ONE protocol + split (+ optional equal fields, e.g. modality='IR').
        Returns {'model', 'experiment', 'value'} or None. Never ranks across protocols."""
        proto = self.protocols.get(protocol_id, {})
        hib = proto.get("higher_is_better", True)
        cands = []
        for m in self.versions(family_id):
            for e in self.experiments_for(m["model_id"]):
                if e.get("protocol_id") != protocol_id or e.get("split") != split or any(e.get(k) != v for k, v in match.items()):
                    continue
                v = (e.get("metrics") or {}).get(metric)
                if isinstance(v, (int, float)):
                    cands.append((v, m, e))
        if not cands:
            return None
        v, m, e = (max if hib else min)(cands, key=lambda c: c[0])
        return {"model": m, "experiment": e, "value": v}

    def demonstration_version(self, family_id: str) -> dict | None:
        """the registered checkpoint of a family, if any (the version used by a registered pass)"""
        reg = [m for m in self.versions(family_id) if m["status"] == "registered"]
        return reg[0] if reg else None

    def same_bytes_paths(self, model: dict) -> list[str]:
        """every physical path holding exactly this model's bytes (its own locations + byte-identical other model ids)"""
        paths = [loc["path"] for loc in model["locations"]]
        for other in model.get("same_bytes_as") or []:
            o = self.by_id.get(other)
            if o:
                paths += [loc["path"] for loc in o["locations"]]
        return list(dict.fromkeys(paths))
