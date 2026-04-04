"""Compatibility wrapper for the StackCube simulation policy loop.

This module keeps the historical `policy_loop_sim.py` entrypoint working while
delegating the actual implementation to `policy_loop_stackcube_sim.py`.
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_MODULE_DIR)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from safety.forewarn.examples.policy_loop_stackcube_sim import (  # noqa: E402
        DummyPolicyCallback,
        PolicyLoopSim,
        main,
    )
else:
    from .policy_loop_stackcube_sim import DummyPolicyCallback, PolicyLoopSim, main

__all__ = ["DummyPolicyCallback", "PolicyLoopSim", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
