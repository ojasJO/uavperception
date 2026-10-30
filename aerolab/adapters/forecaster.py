"""Direction A adapter: Track C learned trajectory forecasters (C1 pooled, C1c domain-aware), VAL trajectories only.

Uses tc_learned.load / tc_learned.predict on samples from tc_forecast.make_samples (causal histories of L = 16 frames, H = 24 future
steps, ground-truth histories = the C1 'oracle history' protocol). Classical references are computed on the same samples:
cp / cv (parameter-free) and dcv with its TRAIN-tuned parameters (runs/aerotrack_temporal/c1/classical_tuning.json).
Metrics in pixels, pooled over visible (sample, horizon) points -- forecasting ADE / NLL, never tracking mSA.
"""
import json
import sys

import numpy as np

from ..paths import REPO
from .base import AdapterError, Compatibility, ModelAdapter, verify_file

TC = REPO / "experiments" / "aerotrack_temporal" / "scripts"
TUNING = REPO / "runs" / "aerotrack_temporal" / "c1" / "classical_tuning.json"


def _sys_path():
    if str(TC) not in sys.path:
        sys.path.insert(0, str(TC))


class TrackCForecasterAdapter(ModelAdapter):
    id = "trackc_forecaster"
    tasks = ("trajectory_forecasting",)

    def __init__(self, registry=None):
        super().__init__()
        self.registry = registry
        self.meta = None

    def _domain_of_model(self, rec):
        run = rec["run_id"]
        for d in ("antiuav_IR", "antiuav_RGB", "cst"):
            if run.startswith("c1c_tfm_perdom_") and run[len("c1c_tfm_perdom_"):].startswith(d):
                return d
        return None

    def can_run(self, model_record, dataset_id, modality=None, domain=None):
        if model_record.get("adapter") != self.id:
            return Compatibility(False, f"{model_record['model_id']} uses adapter {model_record.get('adapter')!r}, not {self.id!r}")
        if not model_record["primary_path"].endswith(".pt"):
            return Compatibility(False, "ONNX exports are deployment artifacts; load the .pt checkpoint of the same forecaster")
        if dataset_id != "trackc_trajectories":
            return Compatibility(False, "forecasters take track histories: choose the dataset 'trackc_trajectories' (Track C VAL trajectories)")
        own = self._domain_of_model(model_record)
        if domain and own and domain != own:
            return Compatibility(False, f"domain-isolated model trained on '{own}' only; the selected sequence is '{domain}'")
        try:
            self._resolve_workspace_path(model_record, self.registry)
        except AdapterError as e:
            return Compatibility(False, str(e))
        return Compatibility(True, "", [])

    def load(self, model_record, device=None):
        self.record = model_record
        c = self.can_run(model_record, "trackc_trajectories")
        if not c.ok:
            raise AdapterError(c.reason)
        t0 = self._timer()
        w = self._resolve_workspace_path(model_record, self.registry)
        verify_file(w, model_record["sha256"])
        self.weights_path = w
        _sys_path()
        import tc_learned
        self.model, self.meta = tc_learned.load(w)
        self.device = "cpu"
        self.load_s = self._timer() - t0
        return self

    def predict(self, traj_seq: dict, max_anchors: int | None = None, keep_anchors: int | None = None, **_):
        """traj_seq: one tc_data.load_split record (VAL). -> forecast result with per-anchor forecasts + ADE vs classical references."""
        if self.model is None:
            raise AdapterError("load() first")
        c = self.can_run(self.record, "trackc_trajectories", domain=traj_seq["domain"])
        if not c.ok:
            raise AdapterError(c.reason)
        _sys_path()
        import tc_forecast as F
        import tc_learned
        m = self.meta
        S = F.make_samples([traj_seq], L=m["L"], H=m["H"])
        if not len(S["t"]):
            raise AdapterError(f"{traj_seq['key']}: no anchor with a visible target at t and t-1 and a visible future")
        if max_anchors and len(S["t"]) > max_anchors:
            idx = np.linspace(0, len(S["t"]) - 1, max_anchors).astype(int)
            S = {k: v[idx] for k, v in S.items()}
        t0 = self._timer()
        mean, var = tc_learned.predict(self.model, S, abs_pos=m["abs_pos"], dom_tok=m.get("dom_tok", False))
        ms = (self._timer() - t0) * 1000 / len(S["t"])
        err = F.point_errors(S, mean)
        d2, nll = F.gauss_terms(S, mean, var)
        res = {"learned": F.summarize(S, err, d2, nll)}
        tuning = json.loads(TUNING.read_text(encoding="utf-8")).get("domains", {}) if TUNING.exists() else {}
        refs = {"cp": F.forecast_classical(S, "cp"), "cv": F.forecast_classical(S, "cv")}
        dcv = ((tuning.get(traj_seq["domain"]) or {}).get("dcv") or {}).get("chosen")
        if dcv:
            refs["dcv"] = F.forecast_classical(S, "dcv", dcv)
        for k, v in refs.items():
            res[k] = F.summarize(S, F.point_errors(S, v))
        k = int(np.argmin(np.abs(S["t"] - np.median(S["t"]))))             # one representative anchor for plotting
        ex = {"t": int(S["t"][k]), "hist_c": S["hist_c"][k].tolist(), "hist_m": S["hist_m"][k].tolist(), "fut_c": S["fut_c"][k].tolist(),
              "fut_m": S["fut_m"][k].tolist(), "mean": mean[k].tolist(), "std": np.sqrt(var[k]).tolist(),
              "refs": {n: v[k].tolist() for n, v in refs.items()}}
        anchors = None
        if keep_anchors:                                                    # evenly spaced anchors kept for the anchor browser
            ki = np.unique(np.linspace(0, len(S["t"]) - 1, min(int(keep_anchors), len(S["t"]))).astype(int))
            anchors = {"t": S["t"][ki].astype(int).tolist(), "hist_c": S["hist_c"][ki].tolist(), "hist_m": S["hist_m"][ki].tolist(),
                       "fut_c": S["fut_c"][ki].tolist(), "fut_m": S["fut_m"][ki].tolist(), "mean": mean[ki].tolist(),
                       "std": np.sqrt(var[ki]).tolist(), "ade_px": np.nanmean(err[ki], axis=1).tolist(),
                       "refs": {n: v[ki].tolist() for n, v in refs.items()},
                       "refs_ade_px": {n: np.nanmean(F.point_errors(S, v)[ki], axis=1).tolist() for n, v in refs.items()}}
        return {"kind": "forecast", "adapter": self.id, "model_id": self.record["model_id"], "checkpoint_sha256": self.record["sha256"],
                "anchors": anchors, "anchor_t_all": S["t"].astype(int).tolist(),
                "checkpoint_path": self.get_metadata()["checkpoint_path"], "dataset_id": "trackc_trajectories", "split": "val",
                "sequence": traj_seq["key"], "domain": traj_seq["domain"], "n_anchors": int(len(S["t"])),
                "history_source": "ground truth (C1 oracle-history protocol)", "metrics": res, "example": ex,
                "timing_ms": {"forecaster_per_anchor": ms}, "device": "cpu",
                "config": {k2: m.get(k2) for k2 in ("arch", "variant", "L", "H", "abs_pos", "dom_tok", "aug", "seed")},
                "classical_params_source": TUNING.relative_to(REPO).as_posix() + " (TRAIN-tuned)", "warnings": list(c.warnings)}
