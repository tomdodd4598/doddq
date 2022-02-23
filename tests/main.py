import src.doddq.qpe as qpe
import src.doddq.helpers as helpers

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.providers import Backend

from scipy import signal


def standard_qpe_test(num_eval_qubits: int, num_state_qubits: int, backend: Backend, shots: int) -> None:
    unitary_circuit = helpers.rand_clifford_circuit(num_state_qubits)
    unitary_circuit.draw(output='mpl', filename='unitary.png')

    helpers.print_circuit_eigensystem(unitary_circuit)

    circuit = qpe.standard_qpe(num_eval_qubits, num_state_qubits, unitary_circuit)
    circuit.add_register(ClassicalRegister(num_eval_qubits, 'c_eval'))

    for i in range(num_eval_qubits):
        circuit.measure(i, i)
    circuit.draw(output='mpl', filename='circuit_qpe.png')

    helpers.histogram_default_job(circuit, backend, shots, 'counts_qpe.png', True)


def rodeo_circuit(
        num_eval_qubits: int,
        hamiltonian: np.ndarray,
        target_energy: float,
        time_stddev: float
) -> QuantumCircuit:
    circuit = qpe.rodeo_qpe_gaussian(num_eval_qubits, 1, hamiltonian, target_energy, time_stddev)
    circuit.add_register(ClassicalRegister(num_eval_qubits, 'c_eval'))

    for i in range(num_eval_qubits):
        circuit.measure(i, i)

    return circuit


def rodeo_count(
        num_eval_qubits: int,
        hamiltonian: np.ndarray,
        target_energy: float,
        time_stddev: float,
        backend: Backend,
        shots: int
) -> int:
    circuit = rodeo_circuit(num_eval_qubits, hamiltonian, target_energy, time_stddev)
    result = helpers.default_job(circuit, backend, shots, False)
    counts = result.get_counts(circuit)
    return counts.get('0' * num_eval_qubits, 0)


def one_qubit_rodeo_qpe(
        num_eval_qubits: int,
        hamiltonian: np.ndarray,
        time_stddev: float,
        min_energy: float,
        max_energy: float,
        energy_sample_num: int,
        narrowing_factor: float,
        max_iterations: int,
        success_threshold: float,
        backend: Backend,
        shots: int
) -> None:
    def rodeo_internal(min_e: float, max_e: float, iteration: int) -> None:
        mean_e = (min_e + max_e) / 2.0
        energy_arr = np.linspace(min_e, max_e, energy_sample_num)
        iteration += 1

        success_arr = helpers.mp_starmap(
            rodeo_count,
            [(
                num_eval_qubits,
                hamiltonian,
                energy,
                time_stddev * iteration,
                backend,
                shots
            ) for energy in energy_arr]
        )

        # print(f"{iteration} ({mean_e}): {success_arr}")

        peaks, _ = signal.find_peaks(success_arr, height=int(shots * success_threshold))

        if iteration < max_iterations:
            range_offset = (max_e - min_e) / (2.0 * (narrowing_factor ** iteration))
            for peak in peaks:
                peak_energy = energy_arr[peak]
                rodeo_internal(peak_energy - range_offset, peak_energy + range_offset, iteration)
        elif len(peaks) > 0:
            print(f"Eigenvalue estimation: {mean_e}")

    rodeo_internal(min_energy, max_energy, 0)

    circuit = rodeo_circuit(num_eval_qubits, hamiltonian, min_energy, time_stddev)
    circuit.draw(output='mpl', filename='circuit_rodeo.png')


def one_qubit_rodeo_qsp(
        num_eval_qubits: int,
        hamiltonian: np.ndarray,
        target_energy: float,
        time_stddev: float,
        backend: Backend,
        shots: int
) -> None:
    circuit = rodeo_circuit(num_eval_qubits, hamiltonian, target_energy, time_stddev)
    result = helpers.default_job(circuit, backend, shots, False)
    counts = result.get_counts(circuit)
    print(counts)


def main():
    # provider = IBMQ.load_account()
    # simulator = provider.get_backend('simulator_statevector')

    # standard_qpe_test(4, 1, simulator, 4096)

    phi = 0.0
    h_0 = helpers.one_qubit_hamiltonian(-0.08496, -0.89134, 0.26536, 0.57205)
    h_1 = helpers.one_qubit_hamiltonian(-0.84537, 0.00673, -0.29354, 0.18477)

    hamiltonian = h_0 + phi * h_1
    helpers.print_matrix_eigensystem(hamiltonian)

    # one_qubit_rodeo_qpe(4, hamiltonian, 4.0, -3.0, 3.0, 40, 4.0, 3, 0.5, None, 1024)
    one_qubit_rodeo_qsp(4, hamiltonian, 1.019, 4.0, None, 1024)


if __name__ == '__main__':
    main()
