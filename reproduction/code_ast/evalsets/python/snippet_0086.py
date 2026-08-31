import numpy as np
from flask import jsonify

from api.services.algorithms.hhl_algorithm import HHLAlgorithm
from api.services.algorithms.qaoa_algorithm import QAOAAlgorithm
from api.services.helper_service import getCircuitCharacteristics, bad_request
from api.model.circuit_response import CircuitResponse

# TODO
def generate_hhl_algorithm(input):
    matrix = input.get("matrix")
    vector = input.get("vector")

    # Check types and dimensions
    matrix_array = np.array(matrix)
    vector_array = np.array(vector)
    if matrix_array.shape[0] != matrix_array.shape[1]:
        return bad_request("Invalid matrix input! Matrix must be square.")
    hermitian = np.allclose(matrix_array, matrix_array.T)
    if not hermitian:
        return bad_request("Invalid matrix input! Matrix must be hermitian.")
    if matrix_array.shape[0] != vector_array.shape[0]:
        return bad_request(
            "Invalid matrix, vector input! Matrix and vector must be of the same dimension."
        )
    if np.log2(matrix_array.shape[0]) % 1 != 0:
        return bad_request("Invalid matrix input! Input matrix dimension must be 2^n.")

    circuit = HHLAlgorithm.create_circuit(matrix, vector)
    return CircuitResponse(
        circuit.qasm(), "algorithm/hhl", circuit.num_qubits, circuit.depth(), input
    )


# TODO
def generate_qaoa_circuit(input):
    adj_matrix = input.get("adj_matrix")
    beta = input.get("beta")
    gamma = input.get("gamma")
    circuit = QAOAAlgorithm.create_circuit(adj_matrix, beta, gamma)
    return CircuitResponse(
        circuit.qasm(), "algorithm/qaoa", circuit.num_qubits, circuit.depth(), input
    )