"""Background execution for the Flight Deck: one worker thread runs queued experiments one at a time (the GPU is shared), the UI
polls job state. A page rerender never starts or repeats a run: only submit() does, and every job has its own experiment id.

State kept per job (in memory; the authoritative record is the experiment directory on disk):
    status       queued | running | completed | partial | failed | cancelled
    live[slug]   latest frame record, sequence, frames done / total, elapsed, device, rolling FPS over the last 2 s (measured)
    trail[(slug, sequence)]   every frame record received so far (for the live timeline)
"""
import collections
import threading
import time
import traceback
import uuid


class Job:
    def __init__(self, kind, cfg):
        self.id = f"job-{uuid.uuid4().hex[:8]}"
        self.kind, self.cfg = kind, cfg
        self.status = "queued"
        self.cancel = threading.Event()
        self.experiment_id = None
        self.manifest = None
        self.error = None
        self.submitted = time.time()
        self.started = None
        self.finished = None
        self.live = {}
        self.trail = collections.defaultdict(list)
        self.events = collections.Counter()
        self.log = collections.deque(maxlen=200)
        self._win = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def on_event(self, e):
        with self.lock:
            self.events[e["type"]] += 1
            if e["type"] == "started":
                self.experiment_id = e["experiment_id"]
            if e["type"] == "frame":
                now = time.perf_counter()
                w = self._win[e["slug"]]
                w.append(now)
                while w and now - w[0] > 2.0:
                    w.popleft()
                fps = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else None
                self.live[e["slug"]] = {**{k: e[k] for k in ("sequence", "record", "frames_done", "frames_total", "elapsed_s", "device", "model_id")},
                                        "rolling_fps": fps, "updated": time.time()}
                self.trail[(e["slug"], e["sequence"])].append(e["record"])
            elif e["type"] != "frame":
                self.log.append((time.strftime("%H:%M:%S"), {k: v for k, v in e.items() if k != "record"}))

    def snapshot(self):
        with self.lock:
            return {"id": self.id, "kind": self.kind, "status": self.status, "experiment_id": self.experiment_id, "error": self.error,
                    "live": {k: dict(v) for k, v in self.live.items()}, "events": dict(self.events), "log": list(self.log),
                    "submitted": self.submitted, "started": self.started, "finished": self.finished, "cfg": self.cfg}


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.queue = collections.deque()
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._worker, name="flightdeck-worker", daemon=True)
        self.thread.start()

    def submit(self, cfg, kind="sot") -> Job:
        j = Job(kind, cfg)
        with self.cv:
            self.jobs[j.id] = j
            self.queue.append(j)
            self.cv.notify()
        return j

    def get(self, job_id) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id):
        j = self.jobs.get(job_id)
        if j:
            j.cancel.set()
            if j.status == "queued":
                with self.cv:
                    if j in self.queue:
                        self.queue.remove(j)
                j.status = "cancelled"

    def active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.status in ("queued", "running")]

    def _worker(self):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait()
                j = self.queue.popleft()
            if j.cancel.is_set():
                j.status = "cancelled"
                continue
            j.status, j.started = "running", time.time()
            try:
                if j.kind == "sot":
                    from .flightdeck import run
                    man = run(j.cfg, on_event=j.on_event, cancel=j.cancel)
                elif j.kind == "forecast":
                    from .forecasting import run_forecast
                    man = run_forecast(j.cfg, on_event=j.on_event, cancel=j.cancel)
                elif j.kind == "benchmark":
                    from .benchmark import run_benchmark
                    man = run_benchmark(j.cfg, on_event=j.on_event, cancel=j.cancel)
                else:
                    raise ValueError(f"unknown job kind {j.kind!r}")
                j.manifest, j.status = man, man["status"]
                j.experiment_id = man["experiment_id"]
            except Exception as e:  # noqa: BLE001 -- the experiment directory holds the traceback too
                j.status, j.error = "failed", f"{type(e).__name__}: {e}"
                j.log.append((time.strftime("%H:%M:%S"), {"type": "error", "trace": traceback.format_exc()[-1500:]}))
            j.finished = time.time()


_MANAGER = None
_LOCK = threading.Lock()


def manager() -> JobManager:
    global _MANAGER
    with _LOCK:
        if _MANAGER is None:
            _MANAGER = JobManager()
        return _MANAGER


def wait(job: Job, timeout=600, poll=0.2) -> Job:
    t0 = time.time()
    while job.status in ("queued", "running") and time.time() - t0 < timeout:
        time.sleep(poll)
    return job
