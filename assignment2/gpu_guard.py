"""Run one assignment job only while the selected GPU has no competing job.

This is a courtesy monitor, not a cluster scheduler or an exclusive reservation.
It never signals another user's process. Interrupted experiment artifacts are
retained; exact CIFAR resumption is not promised.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def gpu_pids():
    output = subprocess.check_output(
        ['nvidia-smi', '--id=0', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    return {int(line.strip()) for line in output.splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes', type=float, default=20)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--allow_shared', action='store_true',
                        help='Allow GPU sharing only when explicitly authorized.')
    parser.add_argument('script', type=Path)
    args = parser.parse_args()
    if args.minutes <= 0 or not args.script.is_file():
        raise ValueError('Supply a positive time budget and an existing script.')
    competing = gpu_pids()
    if competing and not args.allow_shared:
        raise RuntimeError(f'GPU is occupied; no assignment job started. PIDs: {sorted(competing)}')
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0')
    started = time.monotonic()
    status_file = args.log.with_suffix('.status.json')
    status = dict(status='running', script=str(args.script), budget_minutes=args.minutes,
                  allow_shared_gpu=args.allow_shared)
    with args.log.open('x') as log:
        process = subprocess.Popen([sys.executable, '-u', str(args.script)],
                                   stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, env=environment, start_new_session=True)
        status['pid'] = process.pid
        status_file.write_text(json.dumps(status, indent=2))
        reason = None
        while process.poll() is None:
            time.sleep(5)
            # Recheck completion before querying other jobs or applying limits.
            if process.poll() is not None:
                break
            try:
                other = gpu_pids() - {process.pid}
                if other and not args.allow_shared:
                    reason = f'Yielded GPU to competing PIDs {sorted(other)}'
                elif time.monotonic() - started > 60 * args.minutes:
                    reason = 'Prespecified wall-time budget reached'
            except (subprocess.SubprocessError, ValueError) as error:
                reason = f'GPU occupancy could not be verified: {error}'
            if reason:
                # Only terminate the new process group owned by this launcher.
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
                break
        status.update(status='complete' if process.returncode == 0 else 'interrupted_or_failed',
                      returncode=process.returncode, elapsed_seconds=time.monotonic()-started,
                      reason=reason)
        status_file.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)
    if process.returncode != 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
