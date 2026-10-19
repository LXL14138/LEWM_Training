"""Small spawn-based process pool with one owned job per worker.

Only small job/summary dictionaries cross pipes, never images or MuJoCo objects.
The parent alone updates the dataset manifest. Abrupt worker death is an error,
not a silently retried job; shutdown joins/terminates only this pool's children.
"""

from contextlib import contextmanager
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import signal
import threading
import time


CHILD_THREAD_LIMITS = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "LP_NUM_THREADS",
)}


@contextmanager
def child_environment():
    """Set before spawn/import, then restore the parent's original environment."""
    previous = {name: os.environ.get(name) for name in CHILD_THREAD_LIMITS}
    try:
        os.environ.update(CHILD_THREAD_LIMITS)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _stop_workers(slots, stop):
    # A second Ctrl+C must not leave writers alive while metadata is reconciled.
    can_signal = threading.current_thread() is threading.main_thread()
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN) if can_signal else None
    try:
        stop.set()
        for slot in slots:
            try:
                slot["pipe"].send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = time.monotonic() + 10.0
        for slot in slots:
            slot["process"].join(max(0.0, deadline - time.monotonic()))
        for slot in slots:
            process = slot["process"]
            if process.is_alive():
                process.terminate()
        for slot in slots:
            process = slot["process"]
            process.join(2.0)
            if process.is_alive():
                process.kill()
                process.join(2.0)
            slot["pipe"].close()
            if process.is_alive():
                raise RuntimeError(f"Could not stop collection worker {process.pid}")
            process.close()
    finally:
        if can_signal:
            signal.signal(signal.SIGINT, previous)


def run_workers(jobs, worker_target, workers, on_result):
    """Dispatch dynamically; worker_target(pipe, stop) owns a persistent scene.

    The worker sends {status: complete, row: {id: ...}} or {status: error,
    error: ...}. Callback order is completion order, not episode-id order.
    """
    if type(workers) is not int or workers < 1 or not jobs:
        raise ValueError("Need jobs and a positive integer worker count")
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate episode ids")
    context = mp.get_context("spawn")
    stop = context.Event()
    slots, pending = [], iter(jobs)

    def dispatch(slot):
        job = next(pending, None)
        slot["job"] = job
        slot["pipe"].send(job)

    try:
        with child_environment():
            for _ in range(min(workers, len(jobs))):
                parent, child = context.Pipe()
                process = context.Process(target=worker_target, args=(child, stop))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                slots.append({"process": process, "pipe": parent, "job": None})
        for slot in slots:
            dispatch(slot)
        completed = 0
        while completed < len(jobs):
            active = {slot["pipe"]: slot for slot in slots if slot["job"] is not None}
            for pipe in wait(list(active), timeout=0.2):
                slot = active[pipe]
                job_id = slot["job"]["id"]
                try:
                    message = pipe.recv()
                except (EOFError, OSError) as error:
                    raise RuntimeError(f"Worker exited before finishing episode {job_id}") from error
                if message.get("status") != "complete":
                    raise RuntimeError(f"Episode {job_id}: {message.get('error', 'invalid worker result')}")
                row = message["row"]
                if row.get("id") != job_id:
                    raise RuntimeError("Worker returned the wrong episode id")
                on_result(row)
                completed += 1
                dispatch(slot)
            for slot in slots:
                if (slot["job"] is not None and slot["process"].exitcode is not None
                        and not slot["pipe"].poll()):
                    raise RuntimeError(f"Worker exited with code {slot['process'].exitcode} "
                                       f"during episode {slot['job']['id']}")
    finally:
        _stop_workers(slots, stop)
