"""Background simulation jobs: each job is a subprocess running ltemsim_gui.runner.

JobManager queues jobs (default: one at a time, simulations are CPU heavy), streams
their output, estimates progress from the scripts' own progress prints, keeps a
history in the per-user app-data folder and writes the full log next to the results.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from . import paths

QUEUED, RUNNING, FINISHED, FAILED, CANCELLED = "queued", "running", "finished", "failed", "cancelled"
STATUS_ZH = {QUEUED: "排队中", RUNNING: "运行中", FINISHED: "已完成", FAILED: "失败", CANCELLED: "已取消"}


# ----------------------------------------------------------------------------- progress
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
            if self.re_count.search(line):
                self.count += 1
                self.sub = 0.0
                if self.inner is not None:
                    self.inner = ProgressParser({"type": "steady"})
                self.text = f"{min(self.count, self.total)}/{self.total} 完成"
            elif self.count >= self.total:
                return
            elif self.inner is not None:
                self.inner.feed(line)
                self.sub = min(0.99, self.inner.fraction() / 0.9)
                self.text = f"{self.count}/{self.total} 完成 · 当前 {100 * self.sub:.0f}%"
            elif self.re_sub is not None:
                m = self.re_sub.search(line)
                if m:
                    if self.sub_total:
                        a, b = float(m.group(1)), float(self.sub_total)
                    else:
                        a, b = float(m.group(1)), float(m.group(2))
                    self.sub = min(0.99, a / b) if b > 0 else 0.0
                    self.text = f"{self.count}/{self.total} 完成 · 当前 {100 * self.sub:.0f}%"
        elif k == "steady":
            m = self.re_step.search(line)
            if m:
                stage = "dark" if m.group(1) == "dark" else "light"
                t = float(m.group(3))
                te = self.t_end[stage]
                f = math.log10(max(t, self.dt0) / self.dt0) / math.log10(te / self.dt0)
                self.stage_frac[stage] = max(self.stage_frac.get(stage, 0.0), min(0.98, f))
                self.stage = stage
                self.text = ("暗态加偏压" if stage == "dark" else "光照") + f" · t = {t:.1e} s"
            m = self.re_done.search(line)
            if m:
                stage = "dark" if m.group(1) == "dark" else "light"
                self.stage_frac[stage] = 1.0
                self.text = "暗态收敛" if stage == "dark" else "光照稳态收敛 · 作图中"

    def fraction(self):
        if self.kind == "count":
            return min(1.0, (self.count + self.sub) / self.total)
        if self.kind == "steady":
            return 0.45 * self.stage_frac.get("dark", 0.0) + 0.45 * self.stage_frac.get("light", 0.0)
        return None


# ----------------------------------------------------------------------------- job
class Job(QObject):
    changed = Signal()            # status / progress
    output = Signal(str)          # one or more complete lines

    def __init__(self, title: str, kind: str, spec: dict, progress: dict | None = None, job_id: str | None = None):
        super().__init__()
        self.id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.title = title
        self.kind = kind
        self.spec = spec
        self.progress_spec = progress
        self.out = spec.get("out")
        self.status = QUEUED
        self.created = time.time()
        self.started = None
        self.finished = None
        self.exit_code = None
        self.progress = 0.0 if progress else None
        self.progress_text = ""
        self.lines: list[str] = []
        self.after: "Job | None" = None          # start only once this job has finished successfully
        self.proc: QProcess | None = None
        self._buf = ""
        self._parser = ProgressParser(progress)
        self._log_fh = None
        self.log_path = paths.app_data_dir() / "logs" / f"{self.id}.log"

    # -- serialisation ----------------------------------------------------------
    def to_dict(self):
        return dict(id=self.id, title=self.title, kind=self.kind, spec=self.spec, progress_spec=self.progress_spec,
                    status=self.status, created=self.created, started=self.started, finished=self.finished,
                    exit_code=self.exit_code, log_path=str(self.log_path))

    @classmethod
    def from_dict(cls, d):
        j = cls(d["title"], d["kind"], d["spec"], d.get("progress_spec"), job_id=d["id"])
        j.status = d["status"] if d["status"] not in (QUEUED, RUNNING) else CANCELLED
        j.created, j.started, j.finished, j.exit_code = d.get("created"), d.get("started"), d.get("finished"), d.get("exit_code")
        j.log_path = Path(d.get("log_path", j.log_path))
        j.progress = 1.0 if j.status == FINISHED else None
        return j

    # -- state --------------------------------------------------------------
    @property
    def active(self):
        return self.status in (QUEUED, RUNNING)

    def elapsed(self):
        if not self.started:
            return 0.0
        return (self.finished or time.time()) - self.started

    def full_log(self) -> str:
        if self.lines:
            return "\n".join(self.lines)
        try:
            return self.log_path.read_text(errors="replace")
        except OSError:
            return ""

    def _ingest(self, text: str):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self._buf += text
        if "\n" not in self._buf:
            return
        *complete, self._buf = self._buf.split("\n")
        progressed = False
        for line in complete:
            self.lines.append(line)
            if self._log_fh:
                self._log_fh.write(line + "\n")
            before = (self.progress, self._parser.text)
            self._parser.feed(line)
            f = self._parser.fraction()
            if f is not None:
                self.progress = max(self.progress or 0.0, f)
            self.progress_text = self._parser.text
            progressed |= before != (self.progress, self._parser.text)
        if len(self.lines) > 50000:
            del self.lines[:10000]
        if self._log_fh:
            self._log_fh.flush()
        self.output.emit("\n".join(complete))
        if progressed:
            self.changed.emit()


# ----------------------------------------------------------------------------- manager
class JobManager(QObject):
    job_added = Signal(object)
    job_changed = Signal(object)
    job_finished = Signal(object)
    jobs_reset = Signal()

    _instance = None

    @classmethod
    def instance(cls) -> "JobManager":
        if cls._instance is None:
            cls._instance = JobManager()
        return cls._instance

    def __init__(self):
        super().__init__()
        self.jobs: list[Job] = []
        self.max_parallel = 1
        self._history = paths.app_data_dir() / "jobs.json"
        self._load_history()
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()

    # -- history --------------------------------------------------------------
    def _load_history(self):
        try:
            data = json.loads(self._history.read_text())
        except (OSError, ValueError):
            return
        for d in data[-200:]:
            try:
                self.jobs.append(Job.from_dict(d))
            except Exception:
                continue

    def save_history(self):
        try:
            self._history.write_text(json.dumps([j.to_dict() for j in self.jobs[-200:]], indent=1))
        except OSError:
            pass

    def clear_finished(self):
        self.jobs = [j for j in self.jobs if j.active]
        self.save_history()
        self.jobs_reset.emit()

    # -- submission -----------------------------------------------------------
    def submit(self, title: str, kind: str, script: str, func: str | None = "main", args=None, kwargs=None, out=None,
               progress: dict | None = None, argv=None, after: Job | None = None) -> Job:
        """Queue `script:func(*args, **kwargs)` (or `python script *argv` when func is None).

        `out` (repo-relative or absolute) is the result folder; `after` makes the job wait for another one.
        """
        spec = dict(script=script, func=func, args=list(args or []), kwargs=dict(kwargs or {}), title=title)
        if func is None:
            spec["argv"] = [str(a) for a in (argv or [])]
        if out:
            spec["out"] = paths.rel(out)
        job = Job(title, kind, spec, progress)
        job.after = after
        self.jobs.append(job)
        self.job_added.emit(job)
        self.save_history()
        self._schedule()
        return job

    def rerun(self, job: Job) -> Job:
        """Submit the same job again; an unfinished prerequisite (`after`) is waited for or re-run first."""
        after = None
        if job.after is not None:
            if job.after.active:
                after = job.after
            elif job.after.status != FINISHED:
                after = self.rerun(job.after)
        spec = job.spec
        args, kwargs, out = list(spec.get("args") or []), dict(spec.get("kwargs") or {}), spec.get("out")
        new_out = self._fresh_out(job)
        if new_out is not None:
            old = paths.absolute(out).resolve()
            same = lambda v: isinstance(v, str) and v and paths.absolute(v).resolve() == old
            args = [paths.rel(new_out) if same(a) else a for a in args]
            kwargs = {k: (paths.rel(new_out) if same(v) else v) for k, v in kwargs.items()}
            out = new_out
        return self.submit(job.title, job.kind, spec["script"], spec.get("func", "main"), args, kwargs, out,
                           job.progress_spec, argv=spec.get("argv"), after=after)

    @staticmethod
    def _fresh_out(job: Job):
        """A new results/gui_runs folder for re-running a finished job that owns its folder, else None
        (failed / cancelled runs and follow-up jobs such as demod reuse their folder, e.g. to resume)."""
        if job.status != FINISHED or not job.out:
            return None
        out = paths.absolute(job.out).resolve()
        try:
            out.relative_to(paths.GUI_RUNS.resolve())
        except ValueError:
            return None
        from .runner import meta_name
        if meta_name(str(out), job.spec) != "gui_run.json":
            return None
        prefix = re.sub(r"_\d{8}-\d{6}(_\d+)?$", "", out.name) or job.kind
        return paths.new_run_dir(prefix)

    def running(self):
        return [j for j in self.jobs if j.status == RUNNING]

    def active(self):
        return [j for j in self.jobs if j.active]

    def _schedule(self):
        # dependants of failed / cancelled jobs are cancelled too
        for j in self.jobs:
            if j.status == QUEUED and j.after is not None and j.after.status in (FAILED, CANCELLED):
                j.status = CANCELLED
                j.finished = time.time()
                j.progress_text = "前置任务未完成"
                j.changed.emit()
                self.job_changed.emit(j)
        while len(self.running()) < self.max_parallel:
            ready = [j for j in self.jobs if j.status == QUEUED and (j.after is None or j.after.status == FINISHED)]
            if not ready:
                return
            self._start(ready[0])

    def _start(self, job: Job):
        spec_path = paths.app_data_dir() / "specs" / f"{job.id}.json"
        spec_path.write_text(json.dumps(job.spec, indent=1))
        proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("MPLBACKEND", "Agg")
        env.insert("PYTHONPATH", str(paths.ROOT))
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(str(paths.ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda j=job: self._read(j))
        proc.finished.connect(lambda code, st, j=job: self._finished(j, code))
        proc.errorOccurred.connect(lambda err, j=job: self._error(j, err))
        job.proc = proc
        job.status = RUNNING
        job.started = time.time()
        try:
            job._log_fh = open(job.log_path, "w")
        except OSError:
            job._log_fh = None
        proc.start(paths.python_exe(), ["-u", "-m", "ltemsim_gui.runner", str(spec_path)])
        self.save_history()
        job.changed.emit()
        self.job_changed.emit(job)

    def _read(self, job: Job):
        data = bytes(job.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            job._ingest(data)

    def _error(self, job: Job, err):
        if err == QProcess.ProcessError.FailedToStart:
            job._ingest(f"无法启动 Python 进程: {paths.python_exe()}\n")
            self._finished(job, -1)

    def _finished(self, job: Job, code: int):
        if job.status not in (RUNNING,):
            return
        if job.proc is not None:
            rest = bytes(job.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
            if rest:
                job._ingest(rest)
        if job._buf:
            job._ingest("\n")
        job.exit_code = code
        job.finished = time.time()
        if job.status == RUNNING:
            if getattr(job, "_cancel_requested", False):
                job.status = CANCELLED
            else:
                job.status = FINISHED if code == 0 else FAILED
        if job.status == FINISHED:
            job.progress = 1.0
            job.progress_text = "完成"
        if job._log_fh:
            job._log_fh.close()
            job._log_fh = None
        if job.out:
            try:
                out = paths.absolute(job.out)
                if out.is_dir():
                    from .runner import meta_name
                    (out / meta_name(str(out), job.spec, "gui_log", ".txt")).write_text(job.full_log())
            except OSError:
                pass
        job.proc = None
        self.save_history()
        job.changed.emit()
        self.job_changed.emit(job)
        self.job_finished.emit(job)
        self._schedule()

    def cancel(self, job: Job):
        if job.status == QUEUED:
            job.status = CANCELLED
            job.finished = time.time()
            self.save_history()
            job.changed.emit()
            self.job_changed.emit(job)
            self._schedule()
            return
        if job.status == RUNNING and job.proc is not None:
            job._cancel_requested = True
            pid = job.proc.processId()
            try:
                os.kill(pid, signal.SIGINT)          # KeyboardInterrupt in the runner: clean exit
            except OSError:
                job.proc.kill()
            proc = job.proc
            QTimer.singleShot(4000, lambda: proc.state() != QProcess.ProcessState.NotRunning and proc.kill())

    def shutdown(self):
        for j in self.running():
            j._cancel_requested = True
            if j.proc is not None:
                j.proc.kill()
                j.proc.waitForFinished(2000)
        self.save_history()

    def _on_tick(self):
        for j in self.running():
            j.changed.emit()
            self.job_changed.emit(j)
