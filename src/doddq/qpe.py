import src.doddq.helpers as helpers

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HGate, PhaseGate, QFT
from typing import Callable


class GeneralizedQPEInfo:

    def __init__(self, qr_eval: QuantumRegister, qr_state: QuantumRegister, circuit: QuantumCircuit) -> None:
        self.qr_eval = qr_eval
        self.qr_state = qr_state
        self.circuit = circuit


class GeneralizedQPE(QuantumCircuit):

    def __init__(
            self,
            num_eval_qubits: int,
            num_state_qubits: int,
            qc_eval_begin: QuantumCircuit,
            qc_eval_end: QuantumCircuit,
            qc_state_evolution: Callable[[GeneralizedQPEInfo, int], None],
            name: str = 'GeneralizedQPE'
    ) -> None:
        qr_eval = QuantumRegister(num_eval_qubits, 'eval')
        qr_state = QuantumRegister(num_state_qubits, 'state')
        circuit = QuantumCircuit(qr_eval, qr_state, name=name)

        info = GeneralizedQPEInfo(qr_eval, qr_state, circuit)

        circuit.compose(qc_eval_begin, qubits=qr_eval[:], inplace=True)
        for i in range(num_eval_qubits):
            qc_state_evolution(info, i)
        circuit.compose(qc_eval_end, qubits=qr_eval[:], inplace=True)

        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)


def standard_qpe(
        num_eval_qubits: int,
        num_state_qubits: int,
        unitary: QuantumCircuit,
        name: str = 'StandardQPE'
) -> GeneralizedQPE:
    def state_evolution(info: GeneralizedQPEInfo, index: int) -> None:
        controlled_unitary_pow = unitary.power(2 ** index).control()
        info.circuit.compose(controlled_unitary_pow, qubits=[info.qr_eval[index]] + info.qr_state[:], inplace=True)

    qc_eval_begin = QuantumCircuit(num_eval_qubits)
    qc_eval_begin.append(HGate(), [qc_eval_begin.qubits])

    qc_eval_end = QFT(num_eval_qubits, inverse=True, do_swaps=False).reverse_bits()

    return GeneralizedQPE(num_eval_qubits, num_state_qubits, qc_eval_begin, qc_eval_end, state_evolution, name)


def rodeo_qpe(
        num_eval_qubits: int,
        num_state_qubits: int,
        hamiltonian: np.ndarray,
        target_energy: float,
        time_arr: np.ndarray,
        name: str = 'RodeoQPE'
) -> GeneralizedQPE:
    def unitary(index: int) -> QuantumCircuit:
        return helpers.time_evolution_circuit(hamiltonian, time_arr[index])

    def phase(index: int) -> PhaseGate:
        return PhaseGate(target_energy * time_arr[index])

    def state_evolution(info: GeneralizedQPEInfo, index: int) -> None:
        info.circuit.compose(unitary(index).control(), qubits=[info.qr_eval[index]] + info.qr_state[:], inplace=True)
        info.circuit.compose(phase(index), qubits=[index], inplace=True)

    qc_eval_begin = QuantumCircuit(num_eval_qubits)
    qc_eval_begin.append(HGate(), [qc_eval_begin.qubits])

    qc_eval_end = QuantumCircuit(num_eval_qubits)
    qc_eval_end.append(HGate(), [qc_eval_begin.qubits])

    return GeneralizedQPE(num_eval_qubits, num_state_qubits, qc_eval_begin, qc_eval_end, state_evolution, name)


def rodeo_qpe_gaussian(
        num_eval_qubits: int,
        num_state_qubits: int,
        hamiltonian: np.ndarray,
        target_energy: float,
        time_stddev: float,
        name: str = 'RodeoQPE'
) -> GeneralizedQPE:
    time_arr = helpers.rand_gaussian_array(num_eval_qubits, time_stddev)
    return rodeo_qpe(num_eval_qubits, num_state_qubits, hamiltonian, target_energy, time_arr, name)
