
from cirq import ops
from cirq import qutrit
from cirq.qutrit import raw_types
from cirq.qutrit import common_gates
import cirq


class SmallPlusKCarry(raw_types.TernaryLogicGate,
                            ops.CompositeGate):
    def __init__(self, k, carry_in):
        self.k = k
        self.carry_in = carry_in

    def validate_trits(self, trits):
        super().validate_trits(trits)
        assert len(trits) == len(self.k) + 1, 'Gate only operates on |k|+1 qutrits'
        assert len(trits) >= 3, 'Gate only operates on 3 or more qutrits'

    def applied_to_trits(self, trits):
        carry = self.carry_in and trits[0] == 2
        bin_k = int(''.join(map(str, reversed(self.k))), 2)
        if carry:
            bits = list(trits)
            bits[0] = 1
            bin_k |= 1
        else:
            bits = trits
        bin_input = int(''.join(map(str, reversed(bits[:-1]))), 2)
        if (bin_k + bin_input).bit_length() > len(self.k):
            trits[-1] += 1
            trits[-1] %= 3

        return trits

    def default_decompose(self, qubits):
        k = self.k

        if not self.carry_in:
            while k[0] == 0:
                if len(k) == 1:
                    k = []
                    break
                k = k[1:]
                qubits = qubits[1:]

        if len(k) == 1:
            yield qutrit.C1PlusOne(qubits[0], qubits[1])
        elif len(k) > 1:
            forward = tuple(self.gen_forward_circuit(qubits, k))
            yield from forward
            if k[-1]:
                yield qutrit.C01PlusOne(qubits[len(k)-1], qubits[len(k)])
            else:
                yield qutrit.C2PlusOne(qubits[len(k)-1], qubits[len(k)])
            yield from cirq.inverse(forward)

    def gen_forward_circuit(self, qubits, k):
        if self.carry_in:
            if not k[1]:
                if k[0]:
                    yield qutrit.C12PlusOne(qubits[0], qubits[1])
                else:
                    yield qutrit.C2PlusOne(qubits[0], qubits[1])
            else:
                yield qutrit.F01(qubits[1])
                if k[0]:
                    yield qutrit.C0PlusOne(qubits[0], qubits[1])
                else:
                    yield qutrit.C01PlusOne(qubits[0], qubits[1])
        else:
            # Do the first one - special circumstances
            # Always guarenteed to be a '1'
            if not k[1]:
                yield qutrit.C1PlusOne(qubits[0], qubits[1])
            else:
                yield qutrit.F01(qubits[0])
                yield qutrit.F01(qubits[1])
                yield qutrit.C1PlusOne(qubits[0], qubits[1])

            # Do the remainder of the chains
        for i in range(1, len(k)-1):
            if k[i+1] and k[i]:
                yield qutrit.F01(qubits[i+1])
                yield qutrit.C2PlusOne(qubits[i], qubits[i+1])
            elif k[i+1]:
                yield qutrit.F02(qubits[i+1])
                yield qutrit.C2MinusOne(qubits[i], qubits[i+1])
            elif not k[i+1] and k[i]:
                yield qutrit.PlusOne(qubits[i+1])
                yield qutrit.C2MinusOne(qubits[i], qubits[i+1])
            elif not k[i+1] and not k[i]:
                yield qutrit.C2PlusOne(qubits[i], qubits[i+1])
