from __future__ import annotations

import numpy as np

from safety.forewarn.examples.vlm_api_models import decode_payload, encode_payload


def test_encode_decode_payload_roundtrip_numpy():
    payload = {
        "arr": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "nested": [np.array([True, False], dtype=np.bool_), {"x": np.array([5], dtype=np.int64)}],
    }

    encoded = encode_payload(payload)
    decoded = decode_payload(encoded)

    assert isinstance(decoded["arr"], np.ndarray)
    assert decoded["arr"].shape == (2, 2)
    assert decoded["arr"].dtype == np.float32
    assert np.array_equal(decoded["nested"][0], np.array([True, False], dtype=np.bool_))
    assert decoded["nested"][1]["x"].dtype == np.int64


def test_encode_decode_payload_roundtrip_torch_tensor():
    import torch

    payload = {
        "state": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        "action_seq": torch.zeros((2, 4), dtype=torch.float32),
    }

    encoded = encode_payload(payload)
    decoded = decode_payload(encoded)

    assert isinstance(decoded["state"], np.ndarray)
    assert decoded["state"].dtype == np.float32
    assert decoded["state"].shape == (3,)
    assert isinstance(decoded["action_seq"], np.ndarray)
    assert decoded["action_seq"].shape == (2, 4)


