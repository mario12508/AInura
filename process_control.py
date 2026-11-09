"""Launch and stop only subprocess trees created by this application."""
import os
import signal
import subprocess


def launch_options():
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
           'PYTHONDONTWRITEBYTECODE': '1'}
    if os.name == 'nt':
        return {'env': env, 'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'env': env, 'start_new_session': True}


def stop_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == 'nt':
        # terminate() alone leaves retrieval subprocesses holding GPU memory.
        result = subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                                capture_output=True, timeout=10)
        if result.returncode and proc.poll() is None:
            proc.kill()
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

