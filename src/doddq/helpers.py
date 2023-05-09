import matplotlib.pyplot as pyplot
import multiprocessing as mp
import numba
import numpy as np
import qiskit
import qiskit.quantum_info as quantum_info
import qiskit.tools as tools
import scipy

from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import XGate, YGate, ZGate
from qiskit.extensions import HamiltonianGate
from qiskit.providers import Backend
from qiskit.quantum_info import Operator
from qiskit.result import Result
from typing import Any, Callable, Optional

pool_size = max(1, mp.cpu_count() - 1)


def get_mp_pool() -> mp.Pool:
    return mp.Pool(pool_size)


def mp_starmap(function: Callable, *args, chunksize=mp.cpu_count()) -> Any:
    pool = get_mp_pool()
    return pool.starmap(function, *args, chunksize=chunksize)


def index_nlargest(n: int, indices: Any, values: Any) -> list[Any]:
    arr = np.array([values[index] for index in indices])
    result_indices = arr.argsort()[-n:][::-1]
    return [indices[index] for index in result_indices]


def complex_str(c: complex, places: int) -> str:
    return '{re:+0.0{places}f} {sgn} {im}i'.format(
        re=c.real,
        places=places,
        sgn='+' if c.imag >= 0.0 else '-',
        im='{:0.0{places}f}'.format(abs(c.imag), places=places)
    )


def list_underscore_str(values: list[Any]) -> str:
    return '_'.join(str(value) for value in values)


def print_eigvec(eigvec) -> None:
    print('v = [ ', end='')
    print(', '.join(complex_str(elem, 5) for elem in eigvec), end=' ]\n')


def print_hermitian_eigensystem(mat: np.ndarray) -> None:
    eigvals, eigvecs = np.linalg.eigh(mat)
    for i in range(mat.shape[0]):
        print(f'λ = {eigvals[i]:0.05f}, ', end='')
        print_eigvec(eigvecs[:, i])


def print_hermitian_eigenvalues(mat: np.ndarray) -> None:
    for eigval in np.linalg.eigvalsh(mat):
        print(f'λ = {eigval:0.05f}')


def print_hermitian_extremal_eigenvalues(mat: np.ndarray) -> None:
    eigvals = np.linalg.eigvalsh(mat)
    print(f'λ_min = {min(eigvals):0.05f}, λ_max = {max(eigvals):0.05f}')


def print_hermitian_overlaps(mat: np.ndarray, state: np.ndarray) -> None:
    eigvals, eigvecs = np.linalg.eigh(mat)
    for i in range(mat.shape[0]):
        print(f'λ = {eigvals[i]:0.05f}, P = {abs(np.vdot(state, eigvecs[:, i])) ** 2.0}')


def nearest_hermitian_eigentuple(mat: np.ndarray, target_eigenvalue: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    min_diff = abs(eigvals[0] - target_eigenvalue)
    result = eigvals[0], eigvecs[:, 0]
    for i in range(1, mat.shape[0]):
        diff = abs(eigvals[i] - target_eigenvalue)
        if diff < min_diff:
            min_diff = diff
            result = eigvals[i], eigvecs[:, i]
    return result


def ground_hermitian_eigentuple(mat: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(mat)
    min_eigval = eigvals[0]
    result = eigvals[0], eigvecs[:, 0]
    for i in range(1, mat.shape[0]):
        eigval = eigvals[i]
        if eigval < min_eigval:
            min_eigval = eigval
            result = eigval, eigvecs[:, i]
    return result


def print_unitary_eigensystem(mat: np.ndarray) -> None:
    eigvals, eigvecs = scipy.linalg.schur(mat)
    for i in range(mat.shape[0]):
        eigval = eigvals[i]
        theta = np.angle(eigval) / (2 * np.pi) % 1.0
        print(f'exp(2πiθ) = {complex_str(eigval, 5)}, θ = {theta:+0.05f}, ', end='')
        print_eigvec(eigvecs[:, i])


def print_circuit_eigensystem(circuit: QuantumCircuit) -> None:
    print_unitary_eigensystem(Operator(circuit).data)


@numba.njit(numba.int64(numba.int64))
def int_log2(x: int) -> int:
    return np.rint(np.log2(x))


@numba.njit
def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    else:
        return vector / norm


@numba.njit
def complex_identity(size: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=numba.complex128)
    for i in numba.prange(size):
        result[i, i] = 1.0 + 0.0j
    return result


@numba.njit
def rand_clifford_circuit(qubits: int) -> QuantumCircuit:
    empty = True
    circuit = None
    while empty:
        circuit = quantum_info.random_clifford(qubits).to_circuit()
        empty = len(circuit.count_ops()) == 0
    return circuit


def recursive_kron(*mats: np.ndarray) -> np.ndarray:
    if len(mats) == 1:
        return mats[0]
    return np.kron(mats[0], recursive_kron(*mats[1:]))


@numba.njit
def identity_kron(mat: np.ndarray, identity_dim: int, left: int, right: int) -> np.ndarray:
    return np.kron(complex_identity(identity_dim ** left), np.kron(mat, complex_identity(identity_dim ** right)))


@numba.njit
def controlled_unitary(unitary: np.ndarray, control_qubits: int) -> np.ndarray:
    identity = complex_identity(unitary.shape[0])

    control_dim = 2 ** control_qubits
    identity_projector = complex_identity(control_dim)
    identity_projector[control_dim - 1, control_dim - 1] = 0.0

    unitary_projector = np.zeros((control_dim, control_dim), dtype=numba.complex128)
    unitary_projector[control_dim - 1, control_dim - 1] = 1.0

    return np.kron(identity, identity_projector) + np.kron(unitary, unitary_projector)


pauli_i_mat = complex_identity(2)
pauli_x_mat = XGate().to_matrix()
pauli_y_mat = YGate().to_matrix()
pauli_z_mat = ZGate().to_matrix()


@numba.njit
def one_qubit_hamiltonian(i: float, x: float, y: float, z: float) -> np.ndarray:
    return i * pauli_i_mat + x * pauli_x_mat + y * pauli_y_mat + z * pauli_z_mat


@numba.njit
def two_qubit_hamiltonian(
        ii: float,
        ix: float,
        iy: float,
        iz: float,
        xi: float,
        xx: float,
        xy: float,
        xz: float,
        yi: float,
        yx: float,
        yy: float,
        yz: float,
        zi: float,
        zx: float,
        zy: float,
        zz: float,
) -> np.ndarray:
    return ii * np.kron(pauli_i_mat, pauli_i_mat) +\
           ix * np.kron(pauli_i_mat, pauli_x_mat) +\
           iy * np.kron(pauli_i_mat, pauli_y_mat) +\
           iz * np.kron(pauli_i_mat, pauli_z_mat) + \
           xi * np.kron(pauli_x_mat, pauli_i_mat) + \
           xx * np.kron(pauli_x_mat, pauli_x_mat) + \
           xy * np.kron(pauli_x_mat, pauli_y_mat) + \
           xz * np.kron(pauli_x_mat, pauli_z_mat) + \
           yi * np.kron(pauli_y_mat, pauli_i_mat) + \
           yx * np.kron(pauli_y_mat, pauli_x_mat) + \
           yy * np.kron(pauli_y_mat, pauli_y_mat) + \
           yz * np.kron(pauli_y_mat, pauli_z_mat) + \
           zi * np.kron(pauli_z_mat, pauli_i_mat) + \
           zx * np.kron(pauli_z_mat, pauli_x_mat) + \
           zy * np.kron(pauli_z_mat, pauli_y_mat) + \
           zz * np.kron(pauli_z_mat, pauli_z_mat)


@numba.njit
def heisenberg_chain_hamiltonian(sites: int, j: float, h: float) -> np.ndarray:
    dim = 2 ** sites
    result = np.zeros((dim, dim), dtype=numba.complex128)
    spin_interaction_x = identity_kron(j * pauli_x_mat, 2, 0, 1) @ identity_kron(pauli_x_mat, 2, 1, 0)
    spin_interaction_y = identity_kron(j * pauli_y_mat, 2, 0, 1) @ identity_kron(pauli_y_mat, 2, 1, 0)
    spin_interaction_z = identity_kron(j * pauli_z_mat, 2, 0, 1) @ identity_kron(pauli_z_mat, 2, 1, 0)
    for i in numba.prange(sites):
        if i < sites - 1:
            result += identity_kron(spin_interaction_x, 2, i, sites - i - 2)
            result += identity_kron(spin_interaction_y, 2, i, sites - i - 2)
            result += identity_kron(spin_interaction_z, 2, i, sites - i - 2)
        result += identity_kron(h * pauli_z_mat, 2, i, sites - i - 1)
    return result


def time_evolution_circuit(hamiltonian: np.ndarray, time: float) -> QuantumCircuit:
    circuit = QuantumCircuit(int_log2(hamiltonian.shape[0]))
    circuit.append(HamiltonianGate(hamiltonian, time), circuit.qubits)
    return circuit


@numba.njit
def rand_gaussian_array(size: int, stddev: float) -> np.ndarray:
    return np.random.normal(0.0, stddev, size)


@numba.njit
def rand_element(array: np.ndarray) -> Any:
    return np.random.choice(array)


def list_histogram(x_values: list, y_values: list, title: str, x_label: str, y_label: str) -> None:
    pyplot.clf()
    pyplot.bar(x_values, y_values, width=0.8 * (max(x_values) - min(x_values)) / len(x_values))
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)


def local_simulation_job(circuit: QuantumCircuit, gpu: bool, shots: int, opt: int, log: bool) -> Result:
    if log:
        print(f'Using local simulation backend...')
    simulator = Aer.get_backend('statevector_simulator')
    simulator.set_options(device='GPU' if gpu else 'CPU')
    circuit = qiskit.transpile(circuit, simulator)
    return simulator.run(circuit, shots=shots, optimization_level=opt, memory=True).result()


def ibmq_job(circuit: QuantumCircuit, backend: Backend, shots: int, opt: int, log: bool) -> Result:
    if log:
        print(f'Using IBM Q backend {backend}...')
    job = qiskit.execute(circuit, backend=backend, shots=shots, optimization_level=opt)
    tools.job_monitor(job)
    return job.result()


def default_job(circuit: QuantumCircuit, backend: Optional[Backend], shots: int, log: bool) -> Result:
    if backend is None:
        return local_simulation_job(circuit, False, shots, 3, log)
    else:
        return ibmq_job(circuit, backend, shots, 3, log)
