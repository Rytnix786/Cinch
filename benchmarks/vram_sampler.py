"""Background VRAM telemetry sampler for benchmark duration."""

from __future__ import annotations

import subprocess
import threading
import time
from typing import List, Optional, Tuple


class VRAMSampler:
    """Background thread sampling GPU VRAM usage over time."""

    def __init__(self, sample_interval_seconds: float = 0.2) -> None:
        self.sample_interval = sample_interval_seconds
        self._samples: List[Tuple[float, float]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _query_vram(self) -> Optional[float]:
        """Query current GPU VRAM usage in MiB."""
        try:
            cmd = [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,nounits,noheader",
            ]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
            return float(output.splitlines()[0])
        except Exception:
            return None

    def _sample_loop(self) -> None:
        """Sampling loop executed in background thread."""
        while self._running:
            vram = self._query_vram()
            if vram is not None:
                self._samples.append((time.time(), vram))
            time.sleep(self.sample_interval)

    def start(self) -> None:
        """Start the background VRAM sampler."""
        self._samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for thread termination."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_peak_mib(self) -> Optional[float]:
        """Return the maximum recorded VRAM in MiB."""
        if not self._samples:
            return None
        return max(s[1] for s in self._samples)

    def get_average_mib(self) -> Optional[float]:
        """Return the average recorded VRAM in MiB."""
        if not self._samples:
            return None
        return sum(s[1] for s in self._samples) / len(self._samples)

    def get_samples(self) -> List[Tuple[float, float]]:
        """Return all recorded (timestamp, vram_mib) tuples."""
        return list(self._samples)
