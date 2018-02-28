# Copyright 2018 Google LLC
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

import pytest

from cirq import ops, ParameterizedValue
from cirq.circuits.circuit import Circuit
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.moment import Moment
from cirq.testing import EqualsTester
from cirq.extension import Extensions


def test_equality():
    a = ops.QubitId()
    b = ops.QubitId()

    eq = EqualsTester()

    # Default is empty. Iterables get listed.
    eq.add_equality_group(Circuit(),
                          Circuit([]), Circuit(()))
    eq.add_equality_group(
        Circuit([Moment()]),
        Circuit((Moment(),)))

    # Equality depends on structure and contents.
    eq.add_equality_group(Circuit([Moment([ops.X(a)])]))
    eq.add_equality_group(Circuit([Moment([ops.X(b)])]))
    eq.add_equality_group(
        Circuit(
            [Moment([ops.X(a)]),
             Moment([ops.X(b)])]))
    eq.add_equality_group(
        Circuit([Moment([ops.X(a), ops.X(b)])]))

    # Big case.
    eq.add_equality_group(
        Circuit([
            Moment([ops.H(a), ops.H(b)]),
            Moment([ops.CZ(a, b)]),
            Moment([ops.H(b)]),
        ]))
    eq.add_equality_group(
        Circuit([
            Moment([ops.H(a)]),
            Moment([ops.CNOT(a, b)]),
        ]))


def test_append_single():
    a = ops.QubitId()

    c = Circuit()
    c.append(())
    assert c == Circuit()

    c = Circuit()
    c.append(ops.X(a))
    assert c == Circuit([Moment([ops.X(a)])])

    c = Circuit()
    c.append([ops.X(a)])
    assert c == Circuit([Moment([ops.X(a)])])


def test_append_multiple():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    c.append([ops.X(a), ops.X(b)], InsertStrategy.NEW)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.X(b)])
    ])

    c = Circuit()
    c.append([ops.X(a), ops.X(b)], InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a), ops.X(b)]),
    ])

    c = Circuit()
    c.append(ops.X(a), InsertStrategy.EARLIEST)
    c.append(ops.X(b), InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a), ops.X(b)]),
    ])


def test_append_strategies():
    a = ops.QubitId()
    b = ops.QubitId()
    stream = [ops.X(a), ops.CZ(a, b), ops.X(b), ops.X(b), ops.X(a)]

    c = Circuit()
    c.append(stream, InsertStrategy.NEW)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
        Moment([ops.X(b)]),
        Moment([ops.X(a)]),
    ])

    c = Circuit()
    c.append(stream, InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
        Moment([ops.X(b), ops.X(a)]),
    ])

    c = Circuit()
    c.append(stream, InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b), ops.X(a)]),
        Moment([ops.X(b)]),
    ])


def test_insert():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()

    c.insert(0, ())
    assert c == Circuit()

    with pytest.raises(IndexError):
        c.insert(-1, ())
    with pytest.raises(IndexError):
        c.insert(1, ())

    c.insert(0, [ops.X(a), ops.CZ(a, b), ops.X(b)])
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
    ])

    with pytest.raises(IndexError):
        c.insert(550, ())

    c.insert(1, ops.H(b), strategy=InsertStrategy.NEW)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.H(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
    ])

    c.insert(0, ops.H(b), strategy=InsertStrategy.EARLIEST)
    assert c == Circuit([
        Moment([ops.X(a), ops.H(b)]),
        Moment([ops.H(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(b)]),
    ])


def test_insert_inline_near_start():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit([
        Moment(),
        Moment(),
    ])

    c.insert(1, ops.X(a), strategy=InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([ops.X(a)]),
        Moment(),
    ])

    c.insert(1, ops.Y(a), strategy=InsertStrategy.INLINE)
    assert c ==Circuit([
        Moment([ops.X(a)]),
        Moment([ops.Y(a)]),
        Moment(),
    ])

    c.insert(0, ops.Z(b), strategy=InsertStrategy.INLINE)
    assert c == Circuit([
        Moment([ops.Z(b)]),
        Moment([ops.X(a)]),
        Moment([ops.Y(a)]),
        Moment(),
    ])


def test_operation_at():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, -1) is None
    assert c.operation_at(a, 102) is None

    c = Circuit([Moment()])
    assert c.operation_at(a, 0) is None

    c = Circuit([Moment([ops.X(a)])])
    assert c.operation_at(b, 0) is None
    assert c.operation_at(a, 1) is None
    assert c.operation_at(a, 0) == ops.X(a)

    c = Circuit([Moment(), Moment([ops.CZ(a, b)])])
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, 1) == ops.CZ(a, b)


def test_next_moment_operating_on():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    assert c.next_moment_operating_on([a]) is None
    assert c.next_moment_operating_on([a], 0) is None
    assert c.next_moment_operating_on([a], 102) is None

    c = Circuit([Moment([ops.X(a)])])
    assert c.next_moment_operating_on([a]) == 0
    assert c.next_moment_operating_on([a], 0) == 0
    assert c.next_moment_operating_on([a, b]) == 0
    assert c.next_moment_operating_on([a], 1) is None
    assert c.next_moment_operating_on([b]) is None

    c = Circuit([
        Moment(),
        Moment([ops.X(a)]),
        Moment(),
        Moment([ops.CZ(a, b)])
    ])

    assert c.next_moment_operating_on([a], 0) == 1
    assert c.next_moment_operating_on([a], 1) == 1
    assert c.next_moment_operating_on([a], 2) == 3
    assert c.next_moment_operating_on([a], 3) == 3
    assert c.next_moment_operating_on([a], 4) is None

    assert c.next_moment_operating_on([b], 0) == 3
    assert c.next_moment_operating_on([b], 1) == 3
    assert c.next_moment_operating_on([b], 2) == 3
    assert c.next_moment_operating_on([b], 3) == 3
    assert c.next_moment_operating_on([b], 4) is None

    assert c.next_moment_operating_on([a, b], 0) == 1
    assert c.next_moment_operating_on([a, b], 1) == 1
    assert c.next_moment_operating_on([a, b], 2) == 3
    assert c.next_moment_operating_on([a, b], 3) == 3
    assert c.next_moment_operating_on([a, b], 4) is None


def test_next_moment_operating_on_distance():
    a = ops.QubitId()

    c = Circuit([
        Moment(),
        Moment(),
        Moment(),
        Moment(),
        Moment([ops.X(a)]),
        Moment(),
    ])

    assert c.next_moment_operating_on([a], 0, max_distance=4) is None
    assert c.next_moment_operating_on([a], 1, max_distance=3) is None
    assert c.next_moment_operating_on([a], 2, max_distance=2) is None
    assert c.next_moment_operating_on([a], 3, max_distance=1) is None
    assert c.next_moment_operating_on([a], 4, max_distance=0) is None

    assert c.next_moment_operating_on([a], 0, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 1, max_distance=4) == 4
    assert c.next_moment_operating_on([a], 2, max_distance=3) == 4
    assert c.next_moment_operating_on([a], 3, max_distance=2) == 4
    assert c.next_moment_operating_on([a], 4, max_distance=1) == 4

    assert c.next_moment_operating_on([a], 5, max_distance=0) is None
    assert c.next_moment_operating_on([a], 1, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 3, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 1, max_distance=500) == 4

    # Huge max distances should be handled quickly due to capping.
    assert c.next_moment_operating_on([a], 5, max_distance=10**100) is None


def test_prev_moment_operating_on():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    assert c.prev_moment_operating_on([a]) is None
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([a], 102) is None

    c = Circuit([Moment([ops.X(a)])])
    assert c.prev_moment_operating_on([a]) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a, b]) == 0
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([b]) is None

    c = Circuit([
        Moment([ops.CZ(a, b)]),
        Moment(),
        Moment([ops.X(a)]),
        Moment(),
    ])

    assert c.prev_moment_operating_on([a], 4) == 2
    assert c.prev_moment_operating_on([a], 3) == 2
    assert c.prev_moment_operating_on([a], 2) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a], 0) is None

    assert c.prev_moment_operating_on([b], 4) == 0
    assert c.prev_moment_operating_on([b], 3) == 0
    assert c.prev_moment_operating_on([b], 2) == 0
    assert c.prev_moment_operating_on([b], 1) == 0
    assert c.prev_moment_operating_on([b], 0) is None

    assert c.prev_moment_operating_on([a, b], 4) == 2
    assert c.prev_moment_operating_on([a, b], 3) == 2
    assert c.prev_moment_operating_on([a, b], 2) == 0
    assert c.prev_moment_operating_on([a, b], 1) == 0
    assert c.prev_moment_operating_on([a, b], 0) is None


def test_prev_moment_operating_on_distance():
    a = ops.QubitId()

    c = Circuit([
        Moment(),
        Moment([ops.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
    ])

    assert c.prev_moment_operating_on([a], max_distance=4) is None
    assert c.prev_moment_operating_on([a], 6, max_distance=4) is None
    assert c.prev_moment_operating_on([a], 5, max_distance=3) is None
    assert c.prev_moment_operating_on([a], 4, max_distance=2) is None
    assert c.prev_moment_operating_on([a], 3, max_distance=1) is None
    assert c.prev_moment_operating_on([a], 2, max_distance=0) is None
    assert c.prev_moment_operating_on([a], 1, max_distance=0) is None
    assert c.prev_moment_operating_on([a], 0, max_distance=0) is None

    assert c.prev_moment_operating_on([a], 6, max_distance=5) == 1
    assert c.prev_moment_operating_on([a], 5, max_distance=4) == 1
    assert c.prev_moment_operating_on([a], 4, max_distance=3) == 1
    assert c.prev_moment_operating_on([a], 3, max_distance=2) == 1
    assert c.prev_moment_operating_on([a], 2, max_distance=1) == 1

    assert c.prev_moment_operating_on([a], 6, max_distance=10) == 1
    assert c.prev_moment_operating_on([a], 6, max_distance=100) == 1
    assert c.prev_moment_operating_on([a], 13, max_distance=500) == 1

    # Huge max distances should be handled quickly due to capping.
    assert c.prev_moment_operating_on([a], 1, max_distance=10**100) is None


def test_clear_operations_touching():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit()
    c.clear_operations_touching([a, b], range(10))
    assert c == Circuit()

    c = Circuit([
        Moment(),
        Moment([ops.X(a), ops.X(b)]),
        Moment([ops.X(a)]),
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment(),
        Moment([ops.X(b)]),
        Moment(),
    ])
    c.clear_operations_touching([a], [1, 3, 4, 6, 7])
    assert c == Circuit([
        Moment(),
        Moment([ops.X(b)]),
        Moment([ops.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment([ops.X(b)]),
        Moment(),
    ])

    c = Circuit([
        Moment(),
        Moment([ops.X(a), ops.X(b)]),
        Moment([ops.X(a)]),
        Moment([ops.X(a)]),
        Moment([ops.CZ(a, b)]),
        Moment(),
        Moment([ops.X(b)]),
        Moment(),
    ])
    c.clear_operations_touching([a, b], [1, 3, 4, 6, 7])
    assert c == Circuit([
        Moment(),
        Moment(),
        Moment([ops.X(a)]),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
        Moment(),
    ])


def test_qubits():
    a = ops.QubitId()
    b = ops.QubitId()

    c = Circuit([
        Moment([ops.X(a)]),
        Moment([ops.X(b)]),
    ])
    assert c.qubits() == {a, b}

    c = Circuit([
        Moment([ops.X(a)]),
        Moment([ops.X(a)]),
    ])
    assert c.qubits() == {a}

    c = Circuit([
        Moment([ops.CZ(a, b)]),
    ])
    assert c.qubits() == {a, b}

    c = Circuit([
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(a)])
    ])
    assert c.qubits() == {a, b}


def test_from_ops():
    a = ops.QubitId()
    b = ops.QubitId()

    actual = Circuit.from_ops(
        ops.X(a),
        [ops.Y(a), ops.Z(b)],
        ops.CZ(a, b),
        ops.X(a),
        [ops.Z(b), ops.Y(a)],
    )

    assert actual == Circuit([
        Moment([ops.X(a)]),
        Moment([ops.Y(a), ops.Z(b)]),
        Moment([ops.CZ(a, b)]),
        Moment([ops.X(a), ops.Z(b)]),
        Moment([ops.Y(a)]),
    ])


def test_to_text_diagram_teleportation_to_diagram():
    ali = ops.NamedQubit('(0, 0)')
    bob = ops.NamedQubit('(0, 1)')
    msg = ops.NamedQubit('(1, 0)')
    tmp = ops.NamedQubit('(1, 1)')

    c = Circuit([
        Moment([ops.H(ali)]),
        Moment([ops.CNOT(ali, bob)]),
        Moment([ops.X(msg)**0.5]),
        Moment([ops.CNOT(msg, ali)]),
        Moment([ops.H(msg)]),
        Moment(
            [ops.MeasurementGate()(msg),
             ops.MeasurementGate()(ali)]),
        Moment([ops.CNOT(ali, bob)]),
        Moment([ops.CNOT(msg, tmp)]),
        Moment([ops.CZ(bob, tmp)]),
    ])

    assert c.to_text_diagram().strip() == """
(0, 0): ───H───@───────────X───────M───@───────────
               │           │           │
(0, 1): ───────X───────────┼───────────X───────Z───
                           │                   │
(1, 0): ───────────X^0.5───@───H───M───────@───┼───
                                           │   │
(1, 1): ───────────────────────────────────X───Z───
    """.strip()
    assert c.to_text_diagram(use_unicode_characters=False).strip() == """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------Z---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---Z---
        """.strip()

    assert c.to_text_diagram(transpose=True,
                             use_unicode_characters=False).strip() == """
(0, 0) (0, 1) (1, 0) (1, 1)
|      |      |      |
H      |      |      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      X^0.5  |
|      |      |      |
X-------------@      |
|      |      |      |
|      |      H      |
|      |      |      |
M      |      M      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      @------X
|      |      |      |
|      Z-------------Z
|      |      |      |
        """.strip()


def test_to_text_diagram_extended_gate():
    q = ops.NamedQubit('(0, 0)')
    q2 = ops.NamedQubit('(0, 1)')
    q3 = ops.NamedQubit('(0, 2)')

    class FGate(ops.Gate):
        def __repr__(self):
            return 'python-object-FGate:arbitrary-digits'

    f = FGate()
    c = Circuit([
        Moment([f.on(q)]),
    ])

    # Fallback to repr without extension.
    diagram = Circuit([
        Moment([f.on(q)]),
    ]).to_text_diagram(use_unicode_characters=False)
    assert diagram.strip() == """
(0, 0): ---python-object-FGate:arbitrary-digits---
        """.strip()

    # When used on multiple qubits, show the qubit order as a digit suffix.
    diagram = Circuit([
        Moment([f.on(q, q3, q2)]),
    ]).to_text_diagram(use_unicode_characters=False)
    assert diagram.strip() == """
(0, 0): ---python-object-FGate:arbitrary-digits:0---
           |
(0, 1): ---python-object-FGate:arbitrary-digits:2---
           |
(0, 2): ---python-object-FGate:arbitrary-digits:1---
            """.strip()

    # Succeeds with extension.
    class FGateAsAscii(ops.AsciiDiagrammableGate):
        def __init__(self, f_gate):
            self.f_gate = f_gate

        def ascii_wire_symbols(self):
            return 'F'

    diagram = c.to_text_diagram(Extensions({
        ops.AsciiDiagrammableGate: {
           FGate: FGateAsAscii
       }
    }), use_unicode_characters=False)

    assert diagram.strip() == """
(0, 0): ---F---
        """.strip()


def test_to_text_diagram_parameterized_value():
    q = ops.NamedQubit('cube')

    class PGate(ops.AsciiDiagrammableGate):
        def __init__(self, val):
            self.val = val

        def ascii_wire_symbols(self):
            return 'P',

        def ascii_exponent(self):
            return self.val

    c = Circuit.from_ops(
        PGate(1).on(q),
        PGate(2).on(q),
        PGate(ParameterizedValue('a')).on(q),
        PGate(ParameterizedValue('a', 1)).on(q),
        PGate(ParameterizedValue('%$&#*(')).on(q),
        PGate(ParameterizedValue('%$&#*(', 1)).on(q),
    )
    assert str(c).strip() in [
        "cube: ───P───P^2───P^a───P^(1+a)───P^param('%$&#*(')───P^(1+param('%$"
        "&#*('))───",

        "cube: ───P───P^2───P^a───P^(1+a)───P^param(u'%$&#*(')───P^(1+param(u'"
        "%$&#*('))───",
    ]


def test_to_text_diagram_custom_order():
    qa = ops.NamedQubit('2')
    qb = ops.NamedQubit('3')
    qc = ops.NamedQubit('4')

    c = Circuit([Moment([ops.X(qa), ops.X(qb), ops.X(qc)])])
    diagram = c.to_text_diagram(qubit_order_key=lambda e: int(str(e)) % 3,
                                use_unicode_characters=False)
    assert diagram.strip() == """
3: ---X---

4: ---X---

2: ---X---
    """.strip()
