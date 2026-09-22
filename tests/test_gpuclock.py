"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

The GPU clock keeper: when it is wanted, and that the child starts and stops.
"""
from __future__ import annotations

import sys

from jetlink.server import gpuclock as G


def test_wanted_follows_the_setting_then_the_platform(monkeypatch):
  assert G.wanted('on', 'cpu')
  assert not G.wanted('off', 'coreml-Apple_M3_Max')
  monkeypatch.setattr(sys, 'platform', 'darwin')
  assert G.wanted('auto', 'coreml-Apple_M3_Max')
  assert G.wanted('auto', 'ane-Apple_M3_Max')
  assert not G.wanted('auto', 'cuda')
  monkeypatch.setattr(sys, 'platform', 'linux')
  assert not G.wanted('auto', 'coreml-Apple_M3_Max')


def test_busy_loop_runs_until_told_to_stop():
  calls = []
  n = G._busy_loop(lambda: calls.append(1), should_stop=lambda: len(calls) >= 5)
  assert n == 5 and len(calls) == 5


def test_keeper_child_is_stopped_on_close(monkeypatch):
  # A child that idles like the real one, without needing a GPU: the module
  # main is replaced by a sleep through -c, the keeper only sees a Popen.
  monkeypatch.setattr(G.subprocess, 'Popen', _Sleeper)
  k = G.GpuClockKeeper()
  assert k.start() and k.alive()
  k.close()
  assert not k.alive() and k.proc is None


class _Sleeper:
  """Stands in for Popen: alive until terminated."""

  def __init__(self, cmd, **kw):
    assert cmd[1:3] == ['-m', 'jetlink.server.gpuclock']
    self.pid = 4242
    self._rc = None

  def poll(self):
    return self._rc

  def terminate(self):
    self._rc = -15

  def kill(self):
    self._rc = -9

  def wait(self, timeout=None):
    return self._rc
