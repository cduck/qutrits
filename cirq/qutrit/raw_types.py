
from cirq import abc
from cirq import ops


class TernaryLogicEffect(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply_to_ternary_state(self, state):
        raise NotImplementedError


class TernaryLogicGateOperation(ops.GateOperation, TernaryLogicEffect):
    def apply_to_ternary_state(self, state):
        trits = [state[q] for q in self.qubits]
        new_trits = self.gate.applied_to_trits(trits)
        assert len(trits) == len(new_trits), 'Number of input and output qutrits must be the same'
        for q, trit in zip(self.qubits, new_trits):
            assert trit in range(3), 'Trit must have value 0, 1, or 2'
            state[q] = trit


class TernaryLogicGate(ops.Gate, metaclass=abc.ABCMeta):
    def validate_trits(self, trits):
        assert all(trit in range(3) for trit in trits), 'Trit must have value 0, 1, or 2'
    @abc.abstractmethod
    def applied_to_trits(self, trits):
        raise NotImplementedError

    def on(self, *qubits):
        op = super().on(*qubits)
        op.__class__ = TernaryLogicGateOperation
        return op
