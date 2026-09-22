"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A session on a handle nobody is talking on ends instead of sitting in
'connected' for good: when the device leaves the bus, or when no hello comes.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

from jetlink.server import session as S
from jetlink.transport.base import LinkTimeout


class SilentTransport:
  """recv only ever times out; alive() answers from a script."""

  def __init__(self, alive):
    self._alive = list(alive)
    self.recvs = 0

  def recv(self, timeout=None):
    self.recvs += 1
    raise LinkTimeout('nothing')

  def alive(self):
    return self._alive.pop(0) if self._alive else True

  def send(self, *a, **k):
    pass

  def close(self):
    pass


def _session(transport, monkeypatch, clock):
  monkeypatch.setattr(S, 'time', SimpleNamespace(monotonic=lambda: clock[0], perf_counter=lambda: clock[0], time=lambda: clock[0]))
  s = S.Session.__new__(S.Session)
  s.t = transport
  s.client = ''
  return s


def test_session_ends_when_the_device_leaves_the_bus(monkeypatch, caplog):
  caplog.set_level(logging.INFO)
  clock = [100.0]
  t = SilentTransport(alive=[True, True, False])
  _session(t, monkeypatch, clock).serve_forever()
  assert t.recvs == 3
  assert 'left the bus' in caplog.text


def test_session_ends_without_a_hello_after_the_deadline(monkeypatch, caplog):
  caplog.set_level(logging.INFO)
  clock = [100.0]
  t = SilentTransport(alive=[])

  def tick():
    clock[0] += 10.0
    return True

  t.alive = tick
  _session(t, monkeypatch, clock).serve_forever()
  assert 'no hello' in caplog.text
  assert clock[0] - 100.0 > S.HELLO_DEADLINE


def test_session_with_a_hello_keeps_waiting_while_the_device_is_there(monkeypatch):
  clock = [100.0]
  # alive stays True; a 'said hello' session must not hit the hello deadline,
  # so the only way out is the device leaving.
  t = SilentTransport(alive=[True] * 50 + [False])

  def alive():
    clock[0] += 10.0
    return t._alive.pop(0)

  t.alive = alive
  s = _session(t, monkeypatch, clock)
  s.client = 'modeld/1'
  s.serve_forever()
  assert t.recvs == 51
