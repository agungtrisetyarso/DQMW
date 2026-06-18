from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerOptions
import numpy as np

service = QiskitRuntimeService(channel="ibm_quantum_platform")
backend = service.backend("ibm_kingston")

print(f"Using backend: {backend.name} (Richer Ansatz + Dynamical Decoupling)")

# ====================== CREATE RICHER VARIATIONAL CIRCUIT ======================
theta0 = Parameter('θ0')
theta1 = Parameter('θ1')
theta2 = Parameter('θ2')
theta3 = Parameter('θ3')
theta4 = Parameter('θ4')
theta5 = Parameter('θ5')

qc = QuantumCircuit(2)
cr = ClassicalRegister(2, 'c')
qc.add_register(cr)

# Layer 1
qc.ry(theta0, 0)
qc.ry(theta1, 1)
qc.rz(theta2, 0)
qc.rz(theta3, 1)
qc.cx(0, 1)

# Layer 2
qc.ry(theta4, 0)
qc.ry(theta5, 1)

# Engineered Dissipation
qc.measure([0, 1], [0, 1])

with qc.if_test((cr[0], 1)):
    qc.x(0)

with qc.if_test((cr[1], 1)):
    qc.x(1)

# Final measurement
qc.measure([0, 1], [0, 1])

print("\nRicher Variational Dissipative Circuit + Dynamical Decoupling:")
print(qc)

# ====================== TRANSPILE ======================
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_circuit = pm.run(qc)

# ====================== ENABLE DYNAMICAL DECOUPLING ======================
options = SamplerOptions(dynamical_decoupling={})
sampler = Sampler(backend, options=options)

# ====================== RUN ON REAL HARDWARE ======================
np.random.seed(42)
param_values = np.random.uniform(0, 2*np.pi, 6).tolist()

print(f"\nUsing random parameter values: {np.round(param_values, 3)}")

job = sampler.run([(isa_circuit, param_values)], shots=2048)

print(f"\n✅ Job submitted (Richer Ansatz + DD)! Job ID: {job.job_id()}")
