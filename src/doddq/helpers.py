import multiprocessing as mp

import numpy as np

import qiskit
import qiskit.quantum_info
import qiskit.tools
import qiskit.visualization

import scipy

from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.providers import Backend
from qiskit.quantum_info import Operator
from qiskit.result import Result

from typing import Any


def get_mp_pool() -> mp.Pool:
    return mp.Pool(mp.cpu_count())


def mp_starmap(function, *args) -> Any:
    pool = get_mp_pool()
    return pool.starmap(function, *args)


def complex_str(c: complex, places: int) -> str:
    return '{re:+0.05f} {sgn} {im}i'.format(
        re=c.real,
        sgn='+' if c.imag >= 0 else '-',
        im='{:0.0{places}f}'.format(abs(c.imag), places=places)
    )


def print_eigvec(dim, eigvec) -> None:
    print('v = [ ', end='')
    for j in range(dim):
        elem = complex_str(eigvec[j], 5)
        print(elem, end=', ' if j < dim - 1 else ' ], P = [ ')
    for j in range(dim):
        print(f'{(abs(eigvec[j]) ** 2):0.05f}', end=', ' if j < dim - 1 else ' ]\n')


def print_matrix_eigensystem(mat: np.ndarray) -> None:
    eigvals, eigvecs = np.linalg.eig(mat)
    dim = mat.shape[0]
    for i in range(dim):
        print(f'λ = {complex_str(eigvals[i], 5)}, ', end='')
        print_eigvec(dim, eigvecs[i])


def print_unitary_eigensystem(mat: np.ndarray) -> None:
    eigvals, eigvecs = np.linalg.eig(mat)
    dim = mat.shape[0]
    for i in range(dim):
        eigval = eigvals[i]
        theta = np.angle(eigval) / (2 * np.pi) % 1
        print(f'exp(2πiθ) = {complex_str(eigval, 5)}, θ = {theta:+0.05f}, ', end='')
        print_eigvec(dim, eigvecs[i])


def print_circuit_eigensystem(circuit: QuantumCircuit) -> None:
    print_unitary_eigensystem(Operator(circuit).data)


def int_log2(x: int) -> int:
    return x.bit_length() - 1


def rand_clifford_circuit(qubits: int) -> QuantumCircuit:
    empty = True
    circuit = None
    while empty:
        circuit = qiskit.quantum_info.random_clifford(qubits).to_circuit()
        empty = len(circuit.count_ops()) == 0
    return circuit


class OneQubitMatrices:
    i_mat = np.identity(2)
    x_mat = XGate().to_matrix()
    y_mat = YGate().to_matrix()
    z_mat = ZGate().to_matrix()


def one_qubit_hamiltonian(i: float, x: float, y: float, z: float) -> np.ndarray:
    return i * OneQubitMatrices.i_mat +\
           x * OneQubitMatrices.x_mat +\
           y * OneQubitMatrices.y_mat +\
           z * OneQubitMatrices.z_mat


def time_evolution_circuit(hamiltonian: np.ndarray, time: float) -> QuantumCircuit:
    circuit = QuantumCircuit(int_log2(hamiltonian.shape[0]))
    unitary = scipy.linalg.expm(-1j * time * hamiltonian)
    circuit.append(Operator(unitary), [circuit.qubits])
    return circuit


def rand_gaussian_array(size: int, stddev: float) -> np.ndarray:
    return np.random.normal(0, stddev, size)


def rand_element(array: np.ndarray) -> Any:
    return np.random.choice(array)


def ibmq_job(circuit: QuantumCircuit, backend: Backend, shots: int, opt: int, log: bool) -> Result:
    if log:
        print(f'Using IBM Q backend {backend}...')
    job = qiskit.execute(circuit, backend=backend, shots=shots, optimization_level=opt)
    qiskit.tools.job_monitor(job)
    return job.result()


def histogram_ibmq_job(
        circuit: QuantumCircuit,
        backend: Backend,
        shots: int,
        opt: int,
        plot_name: str,
        log: bool
) -> Result:
    result = ibmq_job(circuit, backend, shots, opt, log)
    counts = result.get_counts(circuit)
    hist = qiskit.visualization.plot_histogram(counts)
    hist.savefig(plot_name)
    return result


def local_simulation_job(circuit: QuantumCircuit, gpu: bool, shots: int, opt: int, log: bool) -> Result:
    if log:
        print(f'Using local simulation backend...')
    simulator = Aer.get_backend('aer_simulator_statevector')
    simulator.set_options(device='GPU' if gpu else 'CPU')
    circuit = qiskit.transpile(circuit, simulator)
    return simulator.run(circuit, shots=shots, optimization_level=opt, memory=True).result()


def histogram_local_simulation_job(
        circuit: QuantumCircuit,
        gpu: bool,
        shots: int,
        opt: int,
        plot_name: str,
        log: bool
) -> Result:
    result = local_simulation_job(circuit, gpu, shots, opt, log)
    counts = result.get_counts(circuit)
    hist = qiskit.visualization.plot_histogram(counts)
    hist.savefig(plot_name)
    return result


def default_job(circuit: QuantumCircuit, backend: Backend, shots: int, log: bool) -> Result:
    if backend is None:
        return local_simulation_job(circuit, False, shots, 3, log)
    else:
        return ibmq_job(circuit, backend, shots, 3, log)


def histogram_default_job(circuit: QuantumCircuit, backend: Backend, shots: int, plot_name: str, log: bool) -> Result:
    if backend is None:
        return histogram_local_simulation_job(circuit, False, shots, 3, plot_name, log)
    else:
        return histogram_ibmq_job(circuit, backend, shots, 3, plot_name, log)
