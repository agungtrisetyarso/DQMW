from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(channel="ibm_quantum_platform")
job = service.job("d8pucu6kodhs7382e2sg")

print(f"Job ID: {job.job_id()}")
print(f"Status: {job.status()}")

if job.status() == "DONE":
    result = job.result()
    counts = result[0].data.c.get_counts()
    
    print("\n✅ Job completed (Richer Ansatz + DD)!")
    print("Measurement Results:")
    print(counts)
    
    total = sum(counts.values())
    print("\nSorted Results with Percentages:")
    for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  |{state}⟩ : {count} shots ({count/total*100:.2f}%)")
else:
    print("\n⏳ Job is still running. Please check again later.")
