"""SHA-256 of files with a cache keyed by (repo-relative path, size, mtime_ns): a file whose size or mtime changed is re-hashed.
The cache (results/registry/.hash_cache.json, git-ignored) is an optimisation only; `verify=True` always re-hashes."""
import hashlib
import json
from pathlib import Path

from .paths import REGISTRY, rel

CACHE_FILE = REGISTRY / ".hash_cache.json"


def sha256_file(path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class HashCache:
    def __init__(self, path: Path = CACHE_FILE):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}
        self.hits = self.misses = 0

    def get(self, p, verify=False) -> str:
        p = Path(p)
        st = p.stat()
        key = rel(p)
        e = self.data.get(key)
        if not verify and e and e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns:
            self.hits += 1
            return e["sha256"]
        self.misses += 1
        h = sha256_file(p)
        self.data[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h}
        return h

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.path)
