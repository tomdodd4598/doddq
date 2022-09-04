from doddq import helpers

import numpy as np
import scipy

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from typing import Any


def rodeo_qpe(
        num_cycles: int,
        hamiltonian: np.ndarray,
        initial_state: Any,
        target_energy: float,
        time_arr: np.ndarray
) -> QuantumCircuit:
    obj_qubits = helpers.int_log2(hamiltonian.shape[0])

    qr_anc = QuantumRegister(1, 'qr_anc')
    cr_anc = ClassicalRegister(num_cycles, 'cr_anc')
    qr_obj = QuantumRegister(obj_qubits, 'qr_obj')
    circuit = QuantumCircuit(qr_anc, cr_anc, qr_obj)

    if initial_state is not None:
        circuit.initialize(initial_state, qr_obj)

    for i in range(num_cycles):
        time = time_arr[i]
        circuit.h(qr_anc)
        unitary = helpers.controlled_unitary(scipy.linalg.expm(-1.0j * time * hamiltonian), 1)
        circuit.unitary(unitary, qr_anc[:] + qr_obj[:])
        circuit.p(target_energy * time, qr_anc)
        circuit.h(qr_anc)
        circuit.measure(qr_anc, cr_anc[i])
        circuit.reset(qr_anc)

    return circuit


def rodeo_qsp(
        num_cycles: int,
        hamiltonian: np.ndarray,
        initial_state: Any,
        expected_state: Any,
        target_energy: float,
        time_arr: np.ndarray
) -> QuantumCircuit:
    obj_qubits = helpers.int_log2(hamiltonian.shape[0])

    qr_anc = QuantumRegister(1, 'qr_anc')
    cr_anc = ClassicalRegister(num_cycles, 'cr_anc')
    qr_obj = QuantumRegister(obj_qubits, 'qr_obj')
    qr_copy = QuantumRegister(obj_qubits, 'qr_copy')
    qr_swap = QuantumRegister(1, 'qr_swap')
    cr_swap = ClassicalRegister(1, 'cr_swap')
    circuit = QuantumCircuit(qr_anc, cr_anc, qr_obj, qr_copy, qr_swap, cr_swap)

    if initial_state is not None:
        circuit.initialize(initial_state, qr_obj)

    for i in range(num_cycles):
        time = time_arr[i]
        circuit.h(qr_anc)
        unitary = helpers.controlled_unitary(scipy.linalg.expm(-1.0j * time * hamiltonian), 1)
        circuit.unitary(unitary, qr_anc[:] + qr_obj[:])
        circuit.p(target_energy * time, qr_anc)
        circuit.h(qr_anc)
        circuit.measure(qr_anc, cr_anc[i])
        circuit.reset(qr_anc)

    circuit.initialize(expected_state, qr_copy)
    circuit.h(qr_swap)
    for i in range(obj_qubits):
        circuit.cswap(qr_swap, qr_obj[i], qr_copy[i])
    circuit.h(qr_swap)
    circuit.measure(qr_swap, cr_swap)

    return circuit
