"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

Keep the Mac GPU clocked up while the model idles between frames.

A 20 ms model at 20 Hz leaves the GPU idle 60% of the time, and after about
20 s of that macOS drops the GPU from 1345 MHz to its 340 MHz floor: every
frame then takes 65 ms, and at 65 ms the GPU is still 30% idle, so the clock
never comes back. Measured on an M3 Max with powermetrics; High Power Mode
alone does not prevent it, and neither does keeping the Mac on power.

What holds the clock is a command queue that never runs dry. A child process
submits a tiny Metal matmul back to back: the GPU work is microseconds, the
launch overhead is what keeps the queue busy. Measured over 6000 frames on
the same Mac: 42.6% of frames over the 50 ms budget without it, 0% with it on
power (8.8% on battery, so the car needs a charger regardless).

A child, not a thread: tinygrad's Metal context has no business in the
server process, and the busy loop wants a core to itself at the lowest
priority. Not a sleep in the loop: a duty cycle under 100% is exactly the
idle the governor acts on, and read three times as slow once the clock
dropped. Only the launch rate matters, so the matrix is small.

ponytail: tinygrad only, so this is Apple silicon in practice (the CUDA and
TensorRT hosts show no such governor). Another Metal path can replace
_busy_loop without touching the keeper.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

log = logging.getLogger('jetlink.gpuclock')

# Big enough for a real kernel launch, small enough that the GPU work stays
# below the launch overhead; 256 and 768 measured the same.
MATRIX = 256
NICENESS = 19


def wanted(setting: str, device: str) -> bool:
  """Is the keeper on for this run? auto = CoreML on a Mac, where it was measured."""
  if setting == 'on':
    return True
  if setting == 'off':
    return False
  return sys.platform == 'darwin' and device.startswith(('coreml', 'ane'))


class GpuClockKeeper:
  """The busy child, started with the server and stopped with it."""

  def __init__(self, python: str = sys.executable):
    self.python = python
    self.proc: subprocess.Popen | None = None

  def start(self) -> bool:
    try:
      self.proc = subprocess.Popen(
        [self.python, '-m', 'jetlink.server.gpuclock'],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
      log.warning("gpu clock keeper did not start: %s", e)
      return False
    log.info("gpu clock keeper running (pid %d)", self.proc.pid)
    return True

  def alive(self) -> bool:
    return self.proc is not None and self.proc.poll() is None

  def close(self) -> None:
    if self.proc is None:
      return
    if self.proc.poll() is None:
      self.proc.terminate()
      try:
        self.proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self.proc.kill()
        self.proc.wait()
    self.proc = None


def _busy_loop(step, should_stop=lambda: False) -> int:
  """Call step() back to back until should_stop(). Returns the iteration count."""
  n = 0
  while not should_stop():
    step()
    n += 1
  return n


def main() -> int:
  try:
    os.nice(NICENESS)
  except (OSError, AttributeError):
    pass
  os.environ.setdefault('DEV', 'METAL')
  try:
    from tinygrad import Tensor
  except ImportError:
    log.warning("no tinygrad, gpu clock keeper idle")
    return 1
  a = Tensor.rand(MATRIX, MATRIX).realize()

  def step():
    nonlocal a
    a = (a @ a * 1e-3).realize()

  try:
    _busy_loop(step)
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == '__main__':
  sys.exit(main())
