"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

A model that carries its own history (comma's export from openpilot 64f9b47 on,
Cinque Terre V3 first): the wire must not change, and every next_X output must
come back as state_X on the next frame.
"""
from __future__ import annotations

import numpy as np

from jetlink.queues import StatefulInputs, make_queues
from jetlink.spec import ModelSpec

# Cinque Terre V3 (f78ed37d/12864), as its ONNX declares it
CTV3M_INPUTS = {
  'new_img': (2, 6, 128, 256),
  'desire': (8,),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'state_img_q': (2, 5, 6, 128, 256),
  'state_desire_q': (132, 1, 8),
  'state_feat_q': (128, 1, 16384),
}
CTV3M_OUTPUTS = {
  'outputs': (1, 18452),
  'next_state_img_q': (2, 5, 6, 128, 256),
  'next_state_desire_q': (132, 1, 8),
  'next_state_feat_q': (128, 1, 16384),
}
DTYPES = {'new_img': np.uint8, 'state_img_q': np.uint8}
# BMRLNAP v4, the last model with host-side queues
BMV4_INPUTS = {
  'img': (1, 12, 128, 256),
  'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 33, 8),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'features_buffer': (1, 32, 32, 512),
}


def spec(inputs, outputs) -> ModelSpec:
  return ModelSpec(sha256='0' * 64, nbytes=1, frame_skip=4, input_shapes=inputs,
                   output_shapes=outputs, output_slices={}, checkpoint=None)


def test_wire_is_unchanged():
  new, old = spec(CTV3M_INPUTS, CTV3M_OUTPUTS), spec(BMV4_INPUTS, {'outputs': (1, 18452)})
  assert new.stateful and not old.stateful
  assert new.state_names == ['state_img_q', 'state_desire_q', 'state_feat_q']
  assert new.model_hw == old.model_hw == (128, 256)
  assert new.warped_shape == old.warped_shape
  assert new.packed_shapes == old.packed_shapes
  assert new.infer_req_nbytes == old.infer_req_nbytes


def test_state_loops_back_and_resets():
  s = spec(CTV3M_INPUTS, CTV3M_OUTPUTS)
  dest = {n: np.zeros(shape, DTYPES.get(n, np.float32)) for n, shape in CTV3M_INPUTS.items()}
  q = make_queues(s, dest)
  assert isinstance(q, StatefulInputs)

  warped = np.full(s.warped_shape, 7, np.uint8)
  packed = np.arange(s.packed_nelem, dtype=np.float32)
  q.step_into(warped, packed, dest)
  assert (dest['new_img'] == 7).all()
  assert dest['desire'].tolist() == list(range(8))
  assert dest['traffic_convention'].tolist() == [[8, 9]]
  assert dest['action_t'].tolist() == [[10, 11]]

  outputs = {n: np.full(shape, 3, np.float32) for n, shape in CTV3M_OUTPUTS.items()}
  q.after_run(outputs)
  for name in s.state_names:
    assert (dest[name] == 3).all(), name
  assert dest['state_img_q'].dtype == np.uint8

  q.reset()
  for name in s.state_names:
    assert not dest[name].any(), name
