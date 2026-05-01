#!/usr/bin/env python3
# constraints.py
"""
Operator-basis constraints / hooks.

Right now this hosts:
- AdinkraConstraint: lightweight, seeded operator-basis generator.

It intentionally stays small and NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from math_utils import hermitianize, fro_norm


def _pauli2() -> Dict[str, np.ndarray]:
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"I": I, "X": X, "Y": Y, "Z": Z}


@dataclass
class AdinkraConstraint:
    """
    Lightweight hook for extra operator 'G' (placeholder).

    Provides an operator basis dict for a given physical dimension d.
    - Always includes I
    - Includes Pauli X/Z embedded in the top-left 2x2 block when d >= 2
    - Adds a seeded random Hermitian operator 'G' normalized to Frobenius norm 1
    """
    seed: int = 0

    def operator_basis(self, d: int) -> Dict[str, np.ndarray]:
        d = int(d)
        rng = np.random.default_rng(int(self.seed) + d * 31)

        ops: Dict[str, np.ndarray] = {}
        ops["I"] = np.eye(d, dtype=np.complex128)

        if d >= 2:
            P = _pauli2()

            X = np.eye(d, dtype=np.complex128)
            Z = np.eye(d, dtype=np.complex128)
            X[:2, :2] = P["X"]
            Z[:2, :2] = P["Z"]
            ops["X"] = X
            ops["Z"] = Z
        else:
            ops["Z"] = np.eye(d, dtype=np.complex128)

        # Seeded Hermitian "extra" operator
        H = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))).astype(np.complex128)
        H = hermitianize(H)
        nH = max(fro_norm(H), 1e-12)
        ops["G"] = (H / nH).astype(np.complex128)

        return ops
