"""Run ANTs registrations in a separate worker process (Aurora 1.9.4+).

ANTsPy keeps Python's global interpreter lock while its C++ registration runs, so a
registration done inside the engine freezes everything else in it for minutes: status
check-ins with the lab, live progress, other tabs, previews, sign-in. Running the same
call in a child Python process keeps the engine answering, and if the system runs out of
memory the kernel stops the worker (it is marked as the first to go), not Aurora.

    from pipeline import antsWorker
    result = antsWorker.registration(fixed=..., moving=..., type_of_transform="SyNOnly", ...)
    result["fwdtransforms"], result["invtransforms"]   # same file lists as ants.registration

Same ANTs build, same arguments, same output files: only the process differs. Images are
sent to the worker as arrays with their spacing/origin/direction (through a pipe, not a
file: /tmp can be RAM-backed), and the worker writes its transforms to its own folder
under the system temp folder, which callers delete as before. The warped images
(warpedmovout/warpedfixout) are not sent back; no caller here uses them (pass
return_images=True if one ever does).

AURORA_ANTS_IN_PROCESS=1 runs registrations in the engine as before (fallback, debugging).
Small 2D slice registrations (resampleAnisotropicVoxels) stay in-process: the 1-2 s it
takes to start a worker would cost more than the call.

Temp files (the system temp folder is RAM on many Linux machines, ~1.5 GB per warp):
  * every worker run folder is named aurora-ants-<engine pid>-*, so a folder whose engine
    is no longer running is known to be stale (sweep_stale_temp: project open and every
    registration). A running engine's folders are never swept, not even empty ones: a
    run folder stays empty until ANTs writes its transforms at the end;
  * remove_temp_outputs(paths) deletes a registration's outputs once the caller has
    copied what it needs;
  * a view wrapped in @temp_scoped removes, when it returns or fails, every run folder
    and every file registered with track_temp_output() during the request. This is the
    safety net for error paths. It replaced an older blanket sweep in elastic
    registration that deleted every *.mat / *.nii.gz in the temp folder, including files
    that belonged to other jobs or programs.
"""

from __future__ import annotations

import contextlib
import functools
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

_AURORA_CLIENT_DIR = Path(__file__).resolve().parent.parent
_TMP_PREFIX = "aurora-ants-"
# This engine's run folders: aurora-ants-<pid>-<random>. Older engines used no pid.
_RUN_DIR_PREFIX = f"{_TMP_PREFIX}{os.getpid()}-"
_RUN_DIR_PID_RE = re.compile(r"^" + re.escape(_TMP_PREFIX) + r"(\d+)-")
# ANTs temp outputs: `<tempfile prefix><index><kind>`, e.g. tmpu47ug2dk1Warp.nii.gz,
# tmpu47ug2dk1InverseWarp.nii.gz, tmpnxzsi9dd0GenericAffine.mat.
_ANTS_TEMP_OUTPUT_RE = re.compile(r"^(?P<prefix>.+?)\d+(?:Warp|InverseWarp|GenericAffine)\.(?:nii\.gz|mat)$")
_IMAGE_TAG = "__aurora_ants_image__"


class AntsWorkerError(RuntimeError):
    """The worker failed (ANTs error, out of memory, crash). The message says which."""


class _RegistrationResult(dict):
    def __missing__(self, key):
        raise KeyError(
            f"{key!r} is not returned by antsWorker.registration; pass return_images=True "
            "to get the warped images back."
        )


# ---------------------------------------------------------------------------
# Images across the process boundary
# ---------------------------------------------------------------------------

def _pack(value):
    """ANTsImage -> plain tuple (array + geometry); lists/tuples/dicts recursively."""
    import ants

    if isinstance(value, ants.ANTsImage):
        has_components = bool(value.has_components)
        # view(): the image's own buffer, no extra copy in the engine while it is sent.
        array = value.numpy() if has_components or not hasattr(value, "view") else value.view()
        return (_IMAGE_TAG, array, tuple(value.origin), tuple(value.spacing),
                value.direction, has_components)
    if isinstance(value, (list, tuple)):
        return type(value)(_pack(item) for item in value)
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    return value


def _unpack(value):
    import ants

    if isinstance(value, tuple) and len(value) == 6 and value[0] == _IMAGE_TAG:
        _, array, origin, spacing, direction, has_components = value
        return ants.from_numpy(array, origin=list(origin), spacing=list(spacing),
                               direction=direction, has_components=has_components)
    if isinstance(value, (list, tuple)):
        return type(value)(_unpack(item) for item in value)
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Engine side
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return True  # exists but not ours (EPERM): keep its folder


def sweep_stale_temp() -> int:
    """
    Remove ANTs worker run folders nobody can still need: ones whose engine (pid in the
    name) is no longer running, e.g. after a kill or a crash in the middle of a
    registration. Folders of a running engine, this one included, are always left alone,
    EVEN WHEN EMPTY: a live run folder stays empty until ANTs writes its transforms at the
    very end of the registration, so removing empty folders deleted folders of runs still
    in progress (another request, a page load or a take-over from a second computer runs
    this sweep) and the worker then failed to write its Warp files after minutes of work.
    A running engine's folders are removed by the request that made them (temp_scope /
    registration). Old-style folders without a pid are removed when empty or a day old.
    Returns the number of folders removed.
    """
    removed = 0
    try:
        entries = list(Path(tempfile.gettempdir()).glob(_TMP_PREFIX + "*"))
    except OSError:
        return 0
    for entry in entries:
        if not entry.is_dir():
            continue
        match = _RUN_DIR_PID_RE.match(entry.name)
        if match:
            stale = not _pid_alive(int(match.group(1)))
        else:
            try:
                entry.rmdir()  # only succeeds when empty
                removed += 1
                continue
            except OSError:
                pass
            try:
                stale = time.time() - entry.stat().st_mtime > 24 * 3600
            except OSError:
                stale = False
        if stale:
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
            print(f"[ANTs worker] removed stale temp folder {entry}", flush=True)
    return removed


_sweep_old_temp_dirs = sweep_stale_temp  # earlier name


def remove_temp_outputs(paths) -> None:
    """
    Delete ANTs temp outputs (and their unreturned siblings, e.g. the InverseWarp of a
    template SyN) once they have been copied where they are needed.

    Only files directly inside the system temp folder, or inside an antsWorker run
    folder (`<temp>/aurora-ants-*/`, one per registration call), are touched.
    """
    temp_dir = os.path.realpath(tempfile.gettempdir())
    to_remove = set()
    worker_dirs = set()
    for path in paths or []:
        if not path or not isinstance(path, str):
            continue
        parent_dir = os.path.dirname(os.path.realpath(path))
        if (
            os.path.dirname(parent_dir) == temp_dir
            and os.path.basename(parent_dir).startswith(_TMP_PREFIX)
        ):
            # Every registration has its own folder; the whole folder belongs to this call.
            worker_dirs.add(parent_dir)
            continue
        if parent_dir != temp_dir:
            continue
        to_remove.add(path)
        match = _ANTS_TEMP_OUTPUT_RE.match(os.path.basename(path))
        if match:
            prefix = match.group("prefix")
            import glob
            for sibling in glob.glob(os.path.join(temp_dir, glob.escape(prefix) + "*")):
                sibling_match = _ANTS_TEMP_OUTPUT_RE.match(os.path.basename(sibling))
                if sibling_match and sibling_match.group("prefix") == prefix:
                    to_remove.add(sibling)
    for path in sorted(to_remove):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"  warning: could not remove ANTs temp output {path}: {exc}")
    for worker_dir in sorted(worker_dirs):
        try:
            shutil.rmtree(worker_dir)
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"  warning: could not remove ANTs worker folder {worker_dir}: {exc}")


def remove_registration_outputs(result) -> None:
    """remove_temp_outputs for every transform file of a registration result."""
    if not result:
        return
    remove_temp_outputs(list(result.get("fwdtransforms") or []) + list(result.get("invtransforms") or []))


# Request-scoped safety net (see the module docstring).
_SCOPE = threading.local()


def track_temp_output(path) -> None:
    """Register a temp file (or run folder) to be removed when the current @temp_scoped view ends."""
    stack = getattr(_SCOPE, "stack", None)
    if stack and path:
        stack[-1].add(str(path))


@contextlib.contextmanager
def temp_scope():
    stack = getattr(_SCOPE, "stack", None)
    if stack is None:
        stack = _SCOPE.stack = []
    stack.append(set())
    try:
        yield
    finally:
        tracked = stack.pop()
        if stack:
            stack[-1] |= tracked  # nested scope: the outer one cleans up
        else:
            remove_temp_outputs(sorted(tracked))


def temp_scoped(view_method):
    """
    Decorator for views that run registrations: their ANTs temp files never outlive the
    request. A failed worker (AntsWorkerError) is answered with a JSON 500 whose "error"
    is the worker's readable reason (out of memory, temp folder deleted, ANTs error), so
    the tool and the job history show it instead of a bare "Request failed (500)".
    """
    @functools.wraps(view_method)
    def wrapper(*args, **kwargs):
        try:
            with temp_scope():
                return view_method(*args, **kwargs)
        except AntsWorkerError as exc:
            traceback.print_exc()
            from rest_framework import status
            from rest_framework.response import Response
            return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    return wrapper


def _describe_exit(returncode: int) -> str:
    if returncode in (-9, 137):
        return ("the system stopped it, most likely because the computer ran out of memory. "
                "Try fewer scans at once, smaller scans, or close other programs")
    if returncode < 0:
        return f"it was stopped by signal {-returncode}"
    return f"it exited with code {returncode}"


def _wait_and_measure(proc) -> dict:
    """Wait for the worker (sleeping, so the engine stays free) and note its peak memory
    and the lowest free memory on the computer, for the terminal log."""
    try:
        import psutil
        child = psutil.Process(proc.pid)
    except Exception:  # noqa: BLE001 - no psutil or already gone: just wait
        proc.wait()
        return {}
    peak = {"worker_gb": 0.0, "min_available_gb": None}
    while proc.poll() is None:
        try:
            rss = child.memory_info().rss / 1024 ** 3
            available = psutil.virtual_memory().available / 1024 ** 3
        except Exception:  # noqa: BLE001 - process ended between checks
            rss, available = 0.0, None
        peak["worker_gb"] = max(peak["worker_gb"], rss)
        if available is not None and (peak["min_available_gb"] is None or available < peak["min_available_gb"]):
            peak["min_available_gb"] = available
        time.sleep(1.0)
    return peak


def _format_peak(peak: dict) -> str:
    if not peak or not peak.get("worker_gb"):
        return ""
    text = f", worker peak memory {peak['worker_gb']:.2f} GB"
    if peak.get("min_available_gb") is not None:
        text += f", lowest free memory {peak['min_available_gb']:.2f} GB"
    return text


def run_stats(result) -> dict:
    """{"peak_gb", "seconds"} of a registration run by the worker ({} when it ran
    in-process). peak_gb is the worker's own peak (ANTs), not the engine's."""
    stats = getattr(result, "stats", None)
    return dict(stats) if isinstance(stats, dict) else {}


_MACHINE = None


def machine_summary() -> dict:
    """The computer a registration ran on (for the report's compute paragraph). Cached."""
    global _MACHINE
    if _MACHINE is not None:
        return dict(_MACHINE)
    import platform
    info = {"os": f"{platform.system()} {platform.release()}".strip(), "cpu": "",
            "logical_cpus": os.cpu_count(), "physical_cpus": None, "ram_gb": None}
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if not info["cpu"] and sys.platform == "darwin":
        try:
            info["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                         capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    if not info["cpu"]:
        info["cpu"] = platform.processor() or ""
    try:
        import psutil
        info["physical_cpus"] = psutil.cpu_count(logical=False)
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:  # noqa: BLE001
        pass
    _MACHINE = info
    return dict(info)


def summarize_runs(runs, fixed_image=None) -> dict:
    """Per-scan compute record for one registration made of one or more worker runs
    (e.g. double pass): peak memory is the highest pass, time the sum. {} when nothing
    was measured (AURORA_ANTS_IN_PROCESS=1)."""
    runs = [run for run in (runs or []) if isinstance(run, dict) and run]
    peaks = [run["peak_gb"] for run in runs if run.get("peak_gb")]
    seconds = [run["seconds"] for run in runs if run.get("seconds")]
    if not peaks and not seconds:
        return {}
    record = {
        "ants_peak_ram_gb": round(max(peaks), 3) if peaks else None,
        "ants_seconds": round(sum(seconds), 1) if seconds else None,
        "passes": len(runs),
        # What the numbers cover, for anyone reading the JSON later.
        "measured": "ANTs worker process: wall-clock time and peak resident memory "
                    "(excludes Aurora's own image loading and saving)",
        "ants_threads": os.environ.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS") or None,
        "machine": machine_summary(),
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        if fixed_image is not None:
            record["grid_shape"] = [int(n) for n in fixed_image.shape]
            record["grid_spacing"] = [float(n) for n in fixed_image.spacing]
    except Exception:  # noqa: BLE001
        pass
    return record


def worker_memory_bytes() -> int:
    """Memory of running ANTs workers (children of this process), for the memory monitor."""
    try:
        import psutil
        return sum(child.memory_info().rss for child in psutil.Process().children(recursive=True))
    except Exception:  # noqa: BLE001
        return 0


def registration(return_images: bool = False, **kwargs) -> dict:
    """ants.registration(**kwargs), run in a worker process. See the module docstring."""
    if os.environ.get("AURORA_ANTS_IN_PROCESS") == "1":
        import ants
        result = ants.registration(**kwargs)
        for path in list(result.get("fwdtransforms") or []) + list(result.get("invtransforms") or []):
            track_temp_output(path)
        return result

    sweep_stale_temp()
    work_dir = tempfile.mkdtemp(prefix=_RUN_DIR_PREFIX)
    track_temp_output(os.path.join(work_dir, "result.pickle"))  # removes the run folder at scope end
    result_path = os.path.join(work_dir, "result.pickle")
    env = dict(os.environ, TMPDIR=work_dir, TEMP=work_dir, TMP=work_dir)
    started = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pipeline.antsWorker", result_path],
            cwd=str(_AURORA_CLIENT_DIR),
            env=env,
            stdin=subprocess.PIPE,
            # ANTs' verbose output goes to the engine's terminal/log, as before.
        )
        try:
            pickle.dump({"kwargs": _pack(kwargs), "return_images": return_images},
                        proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
            proc.stdin.close()
        except BrokenPipeError:
            pass  # the worker died at start; its exit code says why
        peak = _wait_and_measure(proc)
        returncode = proc.returncode
        outcome = None
        try:
            with open(result_path, "rb") as handle:
                outcome = pickle.load(handle)
        except (OSError, EOFError, pickle.UnpicklingError):
            outcome = None
        if returncode != 0 or outcome is None:
            detail = (outcome or {}).get("error") if isinstance(outcome, dict) else None
            if not detail and not os.path.isdir(work_dir):
                # The worker cannot write its transforms or result into a missing folder.
                detail = (f"its temporary folder {work_dir} was deleted while it was running "
                          f"(by another program or a system temp cleanup), so the result "
                          f"could not be saved. Please run the registration again")
            raise AntsWorkerError(
                f"ANTs registration failed: {detail or _describe_exit(returncode)}."
            )
        if outcome.get("error"):
            print(outcome.get("traceback", ""), file=sys.stderr, flush=True)
            raise AntsWorkerError(f"ANTs registration failed: {outcome['error']}")
        result = _RegistrationResult(_unpack(outcome["result"]))
        result.stats = {"peak_gb": round(peak.get("worker_gb") or 0.0, 3) or None,
                        "seconds": round(time.time() - started, 1)}
        print(f"[ANTs worker] registration finished in {time.time() - started:.1f}s"
              f"{_format_peak(peak)}", flush=True)
        return result
    except BaseException:
        # Failed or interrupted: nothing of this run is needed (/tmp may be RAM-backed).
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        try:
            os.remove(result_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Worker side: python -m pipeline.antsWorker <result path>, arguments on stdin
# ---------------------------------------------------------------------------

def _worker_setup() -> None:
    if sys.platform.startswith("linux"):
        # Out of memory: stop this worker before the engine (or anything else).
        try:
            with open("/proc/self/oom_score_adj", "w") as handle:
                handle.write("900")
        except OSError:
            pass
        # End with the engine: no orphan registration eating memory after Aurora quits.
        try:
            import ctypes
            import signal
            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        except Exception:  # noqa: BLE001 - best effort
            pass


def _worker_main(result_path: str) -> int:
    _worker_setup()
    outcome: dict
    try:
        request = pickle.load(sys.stdin.buffer)
        import ants

        kwargs = _unpack(request.pop("kwargs"))  # frees the received arrays as it goes
        result = ants.registration(**kwargs)
        del kwargs
        keep = {"fwdtransforms", "invtransforms"}
        if request.get("return_images"):
            keep |= {"warpedmovout", "warpedfixout"}
        outcome = {"result": _pack({key: value for key, value in result.items() if key in keep})}
    except Exception as exc:  # noqa: BLE001 - reported to the engine
        outcome = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    try:
        with open(result_path, "wb") as handle:
            pickle.dump(outcome, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        # Run folder gone (see sweep_stale_temp): the engine explains it from the exit code.
        print(f"[ANTs worker] could not write the result: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1]))
