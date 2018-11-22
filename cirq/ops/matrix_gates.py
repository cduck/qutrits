# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quantum gates defined by a matrix."""

from typing import Union, cast

import numpy as np

from cirq import linalg, value
from cirq.ops import gate_features, raw_types


def _phase_matrix(turns: float) -> np.ndarray:
    return np.diag([1, np.exp(2j * np.pi * turns)])


class SingleQubitMatrixGate(raw_types.Gate,
                            gate_features.TextDiagrammable,
                            gate_features.PhaseableEffect,
                            gate_features.ExtrapolatableEffect,
                            gate_features.BoundedEffect):
    """A 1-qubit gate defined by its matrix.

    More general than specialized classes like ZGate, but more expensive and
    more float-error sensitive to work with (due to using eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """
        if matrix.shape not in [(2, 2), (3, 3)] or not linalg.is_unitary(matrix):
            raise ValueError('Not a 2x2 or 3x3 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 1:
            raise ValueError(
                'Single-qubit gate applied to multiple qubits: {}({})'.format(
                    self, qubits))

    def extrapolate_effect(self, factor: Union[float, value.Symbol]
                           ) -> 'SingleQubitMatrixGate':
        if isinstance(factor, value.Symbol):
            raise TypeError('SingleQubitMatrixGate cannot be parameterized.')
        e = cast(float, factor)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return SingleQubitMatrixGate(new_mat)

    def trace_distance_bound(self):
        vals = np.linalg.eigvals(self._matrix)
        rotation_angle = abs(np.angle(vals[0] / vals[1]))
        return rotation_angle * 1.2

    def phase_by(self, phase_turns: float, qubit_index: int):
        z = _phase_matrix(phase_turns)
        phased_matrix = z.dot(self._matrix).dot(np.conj(z.T))
        return SingleQubitMatrixGate(phased_matrix)

    def _unitary_(self) -> np.ndarray:
        return self._matrix

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
            wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args),))

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def approx_eq(self, other, ignore_global_phase=True):
        if not isinstance(other, type(self)):
            return NotImplemented
        cmp = (linalg.allclose_up_to_global_phase if ignore_global_phase
               else np.allclose)
        return cmp(self._matrix, other._matrix)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.SingleQubitMatrixGate({})'.format(repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


class TwoQubitMatrixGate(raw_types.Gate,
                         gate_features.TextDiagrammable,
                         gate_features.PhaseableEffect,
                         gate_features.ExtrapolatableEffect):
    """A 2-qubit gate defined only by its matrix.

    More general than specialized classes like CZGate, but more expensive and
    more float-error sensitive to work with (due to using eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """

        if matrix.shape not in [(4, 4), (9, 9)] or not linalg.is_unitary(matrix):
            raise ValueError('Not a 4x4 or 9x9 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 2:
            raise ValueError(
                'Two-qubit gate not applied to two qubits: {}({})'.format(
                    self, qubits))

    def extrapolate_effect(self, factor: Union[float, value.Symbol]
                           ) -> 'TwoQubitMatrixGate':
        if isinstance(factor, value.Symbol):
            raise TypeError('TwoQubitMatrixGate cannot be parameterized.')
        e = cast(float, factor)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return TwoQubitMatrixGate(new_mat)

    def phase_by(self, phase_turns: float, qubit_index: int):
        i = np.eye(2)
        z = _phase_matrix(phase_turns)
        z2 = np.kron(z, i) if qubit_index else np.kron(i, z)
        phased_matrix = z2.dot(self._matrix).dot(np.conj(z2.T))
        return TwoQubitMatrixGate(phased_matrix)

    def approx_eq(self, other, ignore_global_phase=True):
        if not isinstance(other, type(self)):
            return NotImplemented
        cmp = (linalg.allclose_up_to_global_phase if ignore_global_phase
               else np.allclose)
        return cmp(self._matrix, other._matrix)

    def _unitary_(self) -> np.ndarray:
        return self._matrix

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
            wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args), '#2'))

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.TwoQubitMatrixGate({})'.format(repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


def _matrix_to_diagram_symbol(matrix: np.ndarray,
                              args: gate_features.TextDiagramInfoArgs) -> str:
    if args.precision is not None:
        matrix = matrix.round(args.precision)
    result = str(matrix)
    if args.use_unicode_characters:
        lines = result.split('\n')
        for i in range(len(lines)):
            lines[i] = lines[i].replace('[[', '')
            lines[i] = lines[i].replace(' [', '')
            lines[i] = lines[i].replace(']', '')
        w = max(len(line) for line in lines)
        for i in range(len(lines)):
            lines[i] = '│' + lines[i].ljust(w) + '│'
        lines.insert(0, '┌' + ' ' * w + '┐')
        lines.append('└' + ' ' * w + '┘')
        result = '\n'.join(lines)
    return result
