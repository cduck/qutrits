
from cirq import ops
from cirq.qutrit import raw_types, common_gates


def bits_to_val(bits):
    if len(bits) == 0: return 0
    return int(''.join(map(str, reversed(bits))), 2)

def val_to_bits(val, num_bits):
    val &= ~((-1) << num_bits)
    return tuple(map(int, reversed('{:0{}b}'.format(val, num_bits))))


class PlusKGate(raw_types.TernaryLogicGate,
                ops.ReversibleEffect,
                ops.CompositeGate,
                ops.TextDiagrammable):
    def __init__(self, k, *, _inverted=False):
        self.k = k
        self._inverted = _inverted

    def inverse(self):
        return PlusKGate(k=self.k, _inverted=not self._inverted)

    def text_diagram_info(self, args):
        syms = ['[{} k_{}={}]'.format(
                    'Minus' if self._inverted else 'Plus', i, bit)
                for i, bit in enumerate(self.k)]
        return ops.TextDiagramInfo(tuple(syms))

    def validate_trits(self, trits):
        super().validate_trits(trits)
        assert len(trits) == len(self.k), 'Gate only operates on len(k) qutrits'

    def applied_to_trits(self, trits):
        k_val = bits_to_val(self.k)
        reg_val = bits_to_val(trits)
        reg_val += -k_val if self._inverted else k_val
        return val_to_bits(reg_val, len(trits))

    def default_decompose(self, qubits):
        yield from self._gen_all_carry(qubits, self.k, top=True)

        if self.k[0]:
            yield common_gates.F01(qubits[0])

    @classmethod
    def _gen_all_carry(cls, qubits, k, top=True):
        if len(qubits) <= 0: return
        elif len(qubits) == 1:
            return#yield common_gates.PlusOne(qubits[0])
        elif len(qubits) == 2:
            if k[0] == 0:
                yield common_gates.C2F01(qubits[0], qubits[1])
            else:
                yield common_gates.C12F01(qubits[0], qubits[1])
            if k[1]:
                yield common_gates.F01(qubits[1])
        else:
            half = len(qubits) // 2
            yield PlusKCarryGate(k[:half], not top)(*qubits[:half+1])
            yield from cls._gen_all_carry(qubits[:half], k[:half], top)
            yield from cls._gen_all_carry(qubits[half:], k[half:], False)
            yield PlusKUncarryAddGate(k[:half], not top)(*qubits[:half+1])
            if k[half]:
                yield common_gates.F01(qubits[half])


class PlusKCarryGate(raw_types.TernaryLogicGate,
                     ops.ReversibleEffect,
                     ops.CompositeGate,
                     ops.TextDiagrammable):
    def __init__(self, k, carry_in, *, _inverted=False):
        self.k = k
        self.carry_in = carry_in
        self._inverted = _inverted

    def inverse(self):
        return PlusKCarryGate(self.k, carry_in=self.carry_in,
                              _inverted=not self._inverted)

    def text_diagram_info(self, args):
        syms = ['[{}Carry k_{}={}]'.format(
                    '->'*(i==0 and self.carry_in), i, bit)
                for i, bit in enumerate(self.k)]
        syms.append('[-1]' if self._inverted else '[+1]')
        return ops.TextDiagramInfo(tuple(syms))

    def validate_trits(self, trits):
        super().validate_trits(trits)
        assert len(trits) == len(self.k)+1, (
               'Gate only operates on len(k)+1 qutrits')

    def applied_to_trits(self, trits):
        k_val = bits_to_val(self.k)
        if self.carry_in:
            reg_val = bits_to_val(trits[1:-1]) << 1
            k_val |= 1
            reg_val |= trits[0] == 2 or (trits[0] == 1 and self.k[0])
        else:
            reg_val = bits_to_val(trits[:-1])
        if (k_val + reg_val).bit_length() > len(self.k):
            trits[-1] = (trits[-1] + 1 + self._inverted) % 3
        return trits

    def default_decompose(self, qubits):
        if self._inverted:
            op = common_gates.MinusOne(qubits[-1])
        else:
            op = common_gates.PlusOne(qubits[-1])
        return _gen_or_and_control_u(op, qubits[:-1], k, self.carry_in)


class PlusKUncarryAddGate(raw_types.TernaryLogicGate,
                          ops.ReversibleEffect,
                          ops.CompositeGate,
                          ops.TextDiagrammable):
    def __init__(self, k, carry_in, *, _inverted=False):
        self.k = k
        self.carry_in = carry_in
        self._inverted = _inverted

    def inverse(self):
        return self

    def text_diagram_info(self, args):
        syms = ['[{}Uncarry Add k_{}={}]'.format(
                    '->'*(i==0 and self.carry_in), i, bit)
                for i, bit in enumerate(self.k)]
        syms.append('[F02]')
        return ops.TextDiagramInfo(tuple(syms))

    def validate_trits(self, trits):
        super().validate_trits(trits)
        assert len(trits) == len(self.k)+1, (
               'Gate only operates on len(k)+1 qutrits')

    def applied_to_trits(self, trits):
        k_val = bits_to_val(self.k)
        if self.carry_in:
            reg_val = bits_to_val(trits[1:-1]) << 1
            k_val |= 1
            reg_val |= trits[0] == 2 or (trits[0] == 1 and self.k[0])
        else:
            reg_val = bits_to_val(trits[:-1])
        reg_val ^= ~(~0 << (len(trits)-2)) << 1  # Invert controls except first
        if (k_val + reg_val).bit_length() > len(self.k):
            trits[-1] = (trits[-1] * 2 - 1) % 3
        return trits

    def default_decompose(self, qubits):
        op = common_gates.F02(qubits[-1])
        not_ops = [common_gates.F01(q) for q in qubits[1:-1]]
        yield from not_ops
        yield from _gen_or_and_control_u(op, qubits[:-1], k, self.carry_in)
        yield from not_ops


'''class LogAndUpwardGate(raw_types.TernaryLogicGate,
                       ops.ReversibleEffect,
                       ops.CompositeGate,
                       ops.TextDiagrammable):
    def inverse(self):
        ...

    def text_diagram_info(self, args):
        syms = ['[And ]'] * args.known_qubit_count
        syms[0] = '[And>]'
        return ops.TextDiagramInfo(tuple(syms))

    def validate_trits(self, trits):
        super().validate_trits(trits)
        assert len(trits) == len(self.k)+1, (
               'Gate only operates on len(k)+1 qutrits')

    def applied_to_trits(self, trits):
        k_val = bits_to_val(self.k)
        if self.carry_in:
            reg_val = bits_to_val(trits[1:-1]) << 1
            k_val |= 1
            reg_val |= trits[0] == 2 or (trits[0] == 1 and self.k[0])
        else:
            reg_val = bits_to_val(trits[:-1])
        reg_val ^= ~(~0 << (len(trits)-2)) << 1  # Invert controls except first
        if (k_val + reg_val).bit_length() > len(self.k):
            trits[-1] = (trits[-1] * 2 - 1) % 3
        return trits

    def default_decompose(self, qubits):
        op = common_gates.F02(qubits[-1])
        not_ops = [common_gates.F01(q) for q in qubits[1:-1]]
        yield from not_ops
        yield from _gen_or_and_control_u(op, qubits[:-1], k, self.carry_in)
        yield from not_ops'''


def _gen_or_and_control_u(controlled_op, qubits, k, carry_in):
    ...


def _gen_log_and_upward(qubits, qubit_mask=None):
    if qubit_mask is None:
        qubit_mask = {q: True for q in qubits}
    n = len(qubits)
    if n == 0: return
    elif n == 1: return
    elif n == 2:
        assert qubit_mask[qubits[0]]
        qubit_mask[qubits[0]] = False
        yield common_gates.C1PlusOne(qubits[1], qubits[0])
    elif _is_pow_2(n+1):
        yield from _gen_log_and_upward(qubits[1:1+n//2], qubit_mask)
        yield from _gen_log_and_upward(qubits[1+n//2:], qubit_mask)
        assert qubit_mask[qubits[0]]
        qubit_mask[qubits[0]] = False
        yield common_gates.ControlledTernaryGate(common_gates.PlusOne,
                                ((1 if qubit_mask[qubits[1]] else 2,),
                                 (1 if qubit_mask[qubits[1+n//2]] else 2,)))(
                            qubits[1], qubits[1+n//2], qubits[0])
    else:
        nice_n = (1 << ((n+1).bit_length() - 1)) - 1
        yield from _gen_log_and_upward(qubits[1:-nice_n], qubit_mask)
        yield from _gen_log_and_upward(qubits[-nice_n:], qubit_mask)
        assert qubit_mask[qubits[0]]
        qubit_mask[qubits[0]] = False
        if nice_n + 1 == n:
            yield common_gates.ControlledTernaryGate(common_gates.PlusOne,
                                ((1 if qubit_mask[qubits[-nice_n]] else 2,),))(
                            qubits[-nice_n], qubits[0])
        else:
            yield common_gates.ControlledTernaryGate(common_gates.PlusOne,
                                ((1 if qubit_mask[qubits[1]] else 2,),
                                 (1 if qubit_mask[qubits[-nice_n]] else 2,)))(
                            qubits[1], qubits[-nice_n], qubits[0])


def _is_pow_2(n):
    return n == 1 << (n.bit_length() - 1)
