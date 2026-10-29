"""Progress estimates from a job's output lines (no Qt: also used inside the runner process)."""
from __future__ import annotations

import math
import re

_RE_OVERALL = re.compile(r"^@@PAR progress ([0-9.]+)(?: \| (.*))?")


class ProgressParser:
    """Turns output lines into (fraction in [0, 1] or None, short text)."""

    def __init__(self, spec: dict | None):
        self.spec = spec or {}
        self.kind = self.spec.get("type")
        self.count = 0
        self.sub = 0.0
        self.stage = None
        self.stage_frac = {}
        self.text = ""
        self.overall = 0.0
        if self.kind == "count":
            self.re_count = re.compile(self.spec["pattern"])
            self.re_sub = re.compile(self.spec["sub"]) if self.spec.get("sub") else None
            self.sub_total = self.spec.get("sub_total")       # sub regex with one group: value / sub_total
            self.total = max(1, int(self.spec.get("total", 1)))
            self.inner = ProgressParser({"type": "steady"}) if self.spec.get("steady_sub") else None
        elif self.kind == "steady":
            # Newton march prints:  [dark] step  10 t=1.23e-08s ...   and   [light x1] step ...
            self.re_step = re.compile(r"\[(dark|light[^\]]*)\] step\s+(\d+) t=([0-9.eE+-]+)s")
            self.re_done = re.compile(r"\[(dark|light[^\]]*)\] (final:|steady-state Newton converged)")
            self.dt0 = float(self.spec.get("dt0", 1e-13))
            self.t_end = {"dark": float(self.spec.get("dark_t_end", 30.0)), "light": float(self.spec.get("light_t_end", 10.0))}

    def feed(self, line: str):
        k = self.kind
        if k == "count":
            m = _RE_OVERALL.match(line)
            if m:                                   # aggregated progress of parallel runs (ltemsim_gui.parallel)
                self.overall = max(self.overall, float(m.group(1)))
                if m.group(2):
                    self.text = m.group(2)
                return
            if self.re_count.search(line):
                self.count += 1
                self.sub = 0.0
                if self.inner is not None:
                    self.inner = ProgressParser({"type": "steady"})
                self.text = f"{min(self.count, self.total)}/{self.total} done"
            elif self.count >= self.total:
                return
            elif self.inner is not None:
                self.inner.feed(line)
                self.sub = min(0.99, self.inner.fraction() / 0.9)
                self.text = f"{self.count}/{self.total} done · current {100 * self.sub:.0f}%"
            elif self.re_sub is not None:
                m = self.re_sub.search(line)
                if m:
                    if self.sub_total:
                        a, b = float(m.group(1)), float(self.sub_total)
                    else:
                        a, b = float(m.group(1)), float(m.group(2))
                    self.sub = min(0.99, a / b) if b > 0 else 0.0
                    self.text = f"{self.count}/{self.total} done · current {100 * self.sub:.0f}%"
        elif k == "steady":
            m = self.re_step.search(line)
            if m:
                stage = "dark" if m.group(1) == "dark" else "light"
                t = float(m.group(3))
                te = self.t_end[stage]
                f = math.log10(max(t, self.dt0) / self.dt0) / math.log10(te / self.dt0)
                self.stage_frac[stage] = max(self.stage_frac.get(stage, 0.0), min(0.98, f))
                self.stage = stage
                self.text = ("dark: bias on" if stage == "dark" else "light on") + f" · t = {t:.1e} s"
            m = self.re_done.search(line)
            if m:
                stage = "dark" if m.group(1) == "dark" else "light"
                self.stage_frac[stage] = 1.0
                self.text = "dark state converged" if stage == "dark" else "steady state converged · plotting"

    def fraction(self):
        if self.kind == "count":
            return min(1.0, max((self.count + self.sub) / self.total, self.overall))
        if self.kind == "steady":
            return 0.45 * self.stage_frac.get("dark", 0.0) + 0.45 * self.stage_frac.get("light", 0.0)
        return None
