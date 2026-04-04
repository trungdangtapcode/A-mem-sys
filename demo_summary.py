from agentic_memory.memory_system import AgenticMemorySystem

memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview'
)

# === Note ngan (< 150 tu) => KHONG can summary ===
short_note = "Docker containers provide lightweight virtualization by sharing the host OS kernel."
mid_short = memory_system.add_note(short_note)

mem_short = memory_system.read(mid_short)
print("=== SHORT NOTE ===")
print(f"  Content:  {mem_short.content}")
print(f"  Summary:  {mem_short.summary}")
print(f"  Keywords: {mem_short.keywords}")
print(f"  Tags:     {mem_short.tags}")
print()

# === Note dai (> 150 tu) => CAN summary ===
long_note = """
Kubernetes is an open-source container orchestration platform that automates the deployment,
scaling, and management of containerized applications. Originally developed by Google and now
maintained by the Cloud Native Computing Foundation (CNCF), Kubernetes has become the de facto
standard for container orchestration in production environments.

Key components of Kubernetes include the control plane (API server, scheduler, controller manager,
and etcd), worker nodes (kubelet, kube-proxy, and container runtime), and various abstractions
like Pods, Services, Deployments, and StatefulSets. The control plane makes global decisions
about the cluster, while worker nodes run the actual application workloads.

Kubernetes provides several critical features for production deployments: automatic bin packing
optimizes resource utilization by placing containers based on resource requirements; self-healing
automatically restarts failed containers, replaces them, and reschedules them when nodes die;
horizontal scaling can scale applications up and down based on CPU utilization or custom metrics;
service discovery and load balancing distribute network traffic to stabilize deployments;
automated rollouts and rollbacks progressively roll out changes while monitoring application
health; secret and configuration management handles sensitive information without exposing it
in stack configurations.

For networking, Kubernetes implements a flat network model where every Pod gets its own IP address.
This eliminates the need for explicit links between Pods and means that containers within a Pod
can communicate using localhost. The Container Network Interface (CNI) specification allows
various networking solutions like Calico, Flannel, and Cilium to be plugged in.

Storage in Kubernetes is handled through Persistent Volumes (PV) and Persistent Volume Claims (PVC),
which abstract the underlying storage infrastructure. This allows applications to request storage
resources without knowing the details of the physical storage, supporting various backends
including local storage, NFS, cloud provider volumes (AWS EBS, GCP Persistent Disk, Azure Disk),
and distributed storage systems like Ceph and GlusterFS.
"""

mid_long = memory_system.add_note(long_note.strip())

mem_long = memory_system.read(mid_long)
print("=== LONG NOTE ===")
print(f"  Content length: {len(mem_long.content.split())} words")
print(f"  Summary:  {mem_long.summary}")
print(f"  Keywords: {mem_long.keywords}")
print(f"  Tags:     {mem_long.tags}")
print()

# === Test search - note dai phai tim duoc qua summary ===
print("=== SEARCH TESTS ===")
queries = [
    "container orchestration platform",
    "persistent storage and volumes",
    "network model and CNI",
    "Docker virtualization",
]

for q in queries:
    print(f"\nQuery: \"{q}\"")
    results = memory_system.search(q, k=2)
    for i, r in enumerate(results):
        has_summary = "(has summary)" if memory_system.read(r['id']).summary else "(no summary)"
        print(f"  [{i+1}] {r['content'][:80]}... {has_summary}...{r['id']}")

print("\nDone!")
