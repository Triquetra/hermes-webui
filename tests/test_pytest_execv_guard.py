"""Regression guard for the pytest "hangs at 99% then restarts from 0%" loop.

Root cause documented in tests/conftest.py — daemon threads spawned by
api.updates._schedule_restart() can fire process replacement AFTER
monkeypatch teardown restores the real calls, which re-execs (POSIX) or
spawns-and-hard-exits (Windows) the entire pytest process. The conftest
installs a permanent suite-wide boundary that shadows any late-firing
daemon thread:

  * exec-family calls (os.execv/execve/execl/execle/execlp/execlpe/
    execvp/execvpe) are dropped unconditionally;
  * os._exit / os._Exit are dropped from daemon threads only;
  * subprocess.Popen is dropped from daemon threads on win32 only.

This test pins the boundary so a future conftest refactor can't silently
remove it.
"""
import os
import subprocess
import sys
import threading
import time

import pytest

import tests.conftest as conftest


def test_conftest_installs_permanent_execv_guard():
    """os.execv must be replaced by the conftest's safe no-op wrapper."""
    # The wrapper is named `_pytest_session_safe_execv` in conftest.py.
    # Verify the module attribute now points to that wrapper, not the real
    # libc-bound function.
    assert os.execv.__name__ == '_pytest_session_safe_execv', (
        f"os.execv must be the conftest-installed pytest-safe no-op, but "
        f"resolves to {os.execv!r}. Did a recent conftest refactor remove "
        f"the guard? See conftest.py § 'Permanent process-replacement guard "
        f"for the pytest session' — without it, late-firing _schedule_restart "
        f"daemon threads re-exec pytest and the suite loops forever."
    )


def test_safe_execv_returns_none_does_not_exec():
    """The wrapper must be a true no-op — it must not raise, exec, or block."""
    # Pass deliberately bogus args to confirm the wrapper drops them rather
    # than passing them through to the real execv.
    result = os.execv('/nonexistent/binary/path/that/should/not/be/executed',
                      ['/nonexistent/binary/path/that/should/not/be/executed'])
    assert result is None


def test_entire_exec_family_is_guarded():
    """Every exec-family entry point must be the conftest no-op wrapper."""
    for name in ('execv', 'execve', 'execl', 'execle', 'execlp', 'execlpe',
                 'execvp', 'execvpe'):
        real = getattr(conftest, f'_real_{name}', None)
        if real is None:
            # Platform does not expose this variant (e.g. execve on some
            # builds); nothing to guard.
            continue
        wrapper = getattr(os, name)
        assert wrapper.__name__ == f'_pytest_session_safe_{name}', (
            f"os.{name} must be the conftest-installed pytest-safe no-op, but "
            f"resolves to {wrapper!r}."
        )


def test_os_exit_is_guarded_for_daemon_threads():
    """os._exit must be a no-op from daemon threads (the restart thread)."""
    assert os._exit.__name__ == '_pytest_session_safe_exit', (
        f"os._exit must be the conftest-installed pytest-safe wrapper, but "
        f"resolves to {os._exit!r}."
    )

    result = {}

    def daemon_attempt():
        result["value"] = os._exit(0)

    thread = threading.Thread(target=daemon_attempt, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "daemon os._exit attempt must return, not hang"
    assert result.get("value") is None, (
        "os._exit from a daemon thread must be dropped, not terminate the "
        "pytest process"
    )


def test_os_exit_still_works_from_main_thread():
    """The guard must not break legitimate non-daemon exits.

    pytest-timeout's hard-kill timer thread and multiprocessing fork
    children are non-daemon / main-thread and must keep the real os._exit.
    We cannot call the real os._exit here (it would kill pytest), so pin
    the wrapper's pass-through contract structurally: the wrapper must
    delegate to the captured real function for non-daemon callers.
    """
    assert conftest._real_exit is not None
    # The wrapper is a plain function; its non-daemon branch calls
    # _real_exit. Verify the wiring exists and the wrapper is not a blanket
    # no-op.
    import inspect
    source = inspect.getsource(conftest._pytest_session_safe_exit)
    assert "threading.current_thread().daemon" in source
    assert "return _real_exit(_code)" in source


def test_popen_guard_installed():
    """subprocess.Popen must be the conftest guard class (win32 daemon block)."""
    assert subprocess.Popen is conftest._PytestSessionSafePopen, (
        f"subprocess.Popen must be the conftest-installed pytest-safe guard "
        f"class, but resolves to {subprocess.Popen!r}."
    )
    # The guard must subclass the real Popen so Popen[...] annotations,
    # isinstance checks, and instance methods keep working.
    assert issubclass(subprocess.Popen, conftest._real_Popen)
    # Popen[...] type annotations are evaluated at runtime by some deps
    # (e.g. mcp) — the guard must stay subscriptable.
    assert subprocess.Popen[bytes] is not None


def test_popen_still_spawns_from_main_thread():
    """The guard must not interfere with legitimate main-thread spawns.

    The test-server fixture and every subprocess.run-based test spawn from
    the main thread; the wrapper must pass those through to the real Popen.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_blocked_attempts_are_recorded_for_inspection():
    """Blocked replacement attempts must be inspectable by tests."""
    marker = "/nonexistent/recorded/binary"
    os.execv(marker, [marker, "--flag"])
    # The attempts list is shared suite-wide and daemon threads from earlier
    # tests may append at any moment, so assert by content, not exact count.
    matches = [
        r for r in conftest._PROCESS_REPLACEMENT_ATTEMPTS
        if r["kind"] == "execv" and r.get("exe") == marker
    ]
    assert matches, "the blocked execv attempt must be recorded"
    record = matches[-1]
    assert record["args"] == [marker, "--flag"]
    assert record["thread"]


@pytest.mark.skipif(
    os.name != "nt",
    reason="win32 daemon-thread Popen block is Windows-specific",
)
def test_popen_blocked_from_daemon_thread_on_windows():
    """On win32, Popen from a daemon thread must be dropped and recorded."""
    result = {}
    thread_name = "guard-test-daemon-{}-{}".format(os.getpid(), id(object()))

    def daemon_attempt():
        result["value"] = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.exit(0)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    thread = threading.Thread(target=daemon_attempt, daemon=True, name=thread_name)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "daemon Popen attempt must return, not hang"
    dropped = result.get("value")
    assert isinstance(dropped, subprocess.Popen), (
        "Popen from a daemon thread on win32 must be intercepted by the "
        "guard class"
    )
    # The guard's __init__ bails before the real constructor runs, so the
    # instance is never started: no pid, and poll() raises AttributeError
    # instead of reporting a live process.
    assert not hasattr(dropped, "pid"), (
        "Popen from a daemon thread on win32 must not spawn a real process "
        "(the Windows restart branch of _schedule_restart)"
    )
    with pytest.raises(AttributeError):
        dropped.poll()
    # The attempts list is shared suite-wide and daemon threads from earlier
    # tests may append at any moment, so assert by content, not exact count.
    matches = [
        r for r in conftest._PROCESS_REPLACEMENT_ATTEMPTS
        if r["kind"] == "Popen" and r.get("thread") == thread_name
    ]
    assert matches, "the blocked daemon Popen attempt must be recorded"


def test_schedule_restart_daemon_thread_cannot_replace_process(monkeypatch):
    """End-to-end: a real _schedule_restart daemon thread is neutralized.

    This is the exact fork-bomb scenario from #7195: _schedule_restart
    spawns a daemon thread that (on win32) calls subprocess.Popen +
    os._exit(0) and (on POSIX) os.execv. With the suite-wide boundary
    installed, the thread must wake up, attempt the replacement, and be
    dropped — the pytest process must survive and the attempt must be
    recorded for inspection. No monkeypatching of the replacement calls:
    the boundary itself is what is under test.
    """
    import api.updates as upd

    # Keep the restart thread from doing real work it does not need to do
    # in a test (pycache purge walks the whole repo) and from blocking on
    # stream locks; the replacement call itself is left REAL so the guard
    # must intercept it.
    monkeypatch.setattr(upd, "_purge_agent_pycache", lambda *a, **k: None)
    monkeypatch.setattr(upd, "_wait_until_restart_safe", lambda *a, **k: {"restart_blocked": False})

    before = len(conftest._PROCESS_REPLACEMENT_ATTEMPTS)
    upd._schedule_restart(delay=0.05)
    # Give the daemon thread time to wake, run, and attempt replacement.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if len(conftest._PROCESS_REPLACEMENT_ATTEMPTS) > before:
            break
        time.sleep(0.05)
    # The pytest process is still alive — the boundary held.
    assert len(conftest._PROCESS_REPLACEMENT_ATTEMPTS) > before, (
        "the _schedule_restart daemon thread must attempt a replacement "
        "that the suite-wide boundary records"
    )
    record = conftest._PROCESS_REPLACEMENT_ATTEMPTS[-1]
    assert record["kind"] in ("execv", "os._exit", "Popen"), record
    # The would-be replacement command is inspectable.
    if record["kind"] == "Popen":
        assert record["args"], "the would-be Popen command must be recorded"
    elif record["kind"] == "execv":
        assert record["exe"], "the would-be execv binary must be recorded"


def test_guard_is_idempotent_under_double_import():
    """The boundary must survive pytest's double import of conftest.

    pytest loads tests/conftest.py BOTH as `tests.conftest` (package
    conftest at startup) and as bare `conftest` (test files doing
    `from conftest import ...` with tests/ on sys.path). A second
    execution must reuse the same guard objects — otherwise the second
    module would re-wrap subprocess.Popen with a second class and split
    the recorded attempts across two lists.
    """
    import subprocess as _sp

    sentinel = getattr(_sp, "_HERMES_PYTEST_REPLACEMENT_GUARD", None)
    assert sentinel is not None, (
        "the guard sentinel must be installed on the subprocess module"
    )
    # The sentinel's objects are the ones actually wired into os/subprocess.
    assert _sp.Popen is sentinel["Popen"]
    assert os.execv is sentinel["safe_execv"]
    assert os._exit is sentinel["safe_exit"]
    # The conftest module-level names must alias the same objects.
    assert conftest._PytestSessionSafePopen is sentinel["Popen"]
    assert conftest._PROCESS_REPLACEMENT_ATTEMPTS is sentinel["attempts"]
    assert conftest._pytest_session_safe_execv is sentinel["safe_execv"]
    assert conftest._pytest_session_safe_exit is sentinel["safe_exit"]
