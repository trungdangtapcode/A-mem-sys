from agentic_memory.memory_system import AgenticMemorySystem
import json

memory_system = AgenticMemorySystem(
    model_name='all-MiniLM-L6-v2',
    llm_backend='gemini',
    llm_model='gemini-3-flash-preview'
)

# === Dữ liệu đa dạng để test ===
notes = [
    # Kiến thức kỹ thuật
    "Docker containers provide lightweight virtualization by sharing the host OS kernel, unlike virtual machines which need a full guest OS.",
    "PostgreSQL supports JSONB columns which allow storing and querying semi-structured data with indexing support.",
    "Git rebase rewrites commit history by replaying commits on top of another branch, while merge preserves the original history.",
    "Redis is an in-memory data store often used as cache, message broker, and session store in web applications.",
    "Kubernetes orchestrates container deployments, automatically handling scaling, load balancing, and self-healing of services.",

    # Kinh nghiệm cá nhân
    "Hôm nay tôi học được cách debug memory leak trong Python bằng tracemalloc và objgraph.",
    "Cuộc họp sprint review cho thấy team cần cải thiện test coverage từ 60% lên 80% trước Q3.",
    "Tôi thích dùng Neovim với LazyVim config vì nó nhanh và customizable hơn VS Code cho Python development.",

    # Khoa học và nghiên cứu
    "Transformer architecture uses self-attention mechanism to process sequences in parallel, replacing the sequential nature of RNNs.",
    "RAG (Retrieval Augmented Generation) combines vector search with LLM generation to reduce hallucination and ground responses in facts.",
    "Fine-tuning large language models with LoRA reduces memory requirements by only training low-rank adapter matrices.",
    "Diffusion models generate images by learning to reverse a gradual noising process, producing high-quality outputs.",

    # Cuộc sống và sở thích
    "Cà phê Arabica từ Đà Lạt có vị chua thanh và hương hoa nhẹ, khác biệt với Robusta đắng đậm của Buôn Ma Thuột.",
    "Chạy bộ buổi sáng 5km mỗi ngày giúp tăng năng suất làm việc và giảm stress đáng kể.",
    "Cuốn sách 'Designing Data-Intensive Applications' của Martin Kleppmann là must-read cho backend engineers.",

    # Công việc và dự án
    "API gateway cần rate limiting 1000 req/s per user và circuit breaker pattern để tránh cascade failure.",
    "Migration từ monolith sang microservices nên bắt đầu với strangler fig pattern, tách dần từng bounded context.",
    "CI/CD pipeline hiện tại mất 15 phút, cần optimize bằng cách cache Docker layers và chạy test song song.",

    # Mẹo và tips
    "Dùng Python's functools.lru_cache để cache kết quả hàm tính toán nặng, giảm thời gian xử lý đáng kể.",
    "SSH tunneling với lệnh 'ssh -L 5432:db-host:5432 bastion' giúp truy cập database qua jump server an toàn.",
]

print(f"=== Adding {len(notes)} memories ===\n")
memory_ids = []
for i, note in enumerate(notes):
    mid = memory_system.add_note(note)
    memory_ids.append(mid)
    print(f"[{i+1:2d}] {note[:70]}...")

print(f"\n{'='*60}")
print(f"Total memories stored: {len(memory_ids)}")

# === Xem chi tiết một số memory ===
print(f"\n{'='*60}")
print("=== Memory details (5 samples) ===\n")
for idx in [0, 4, 8, 12, 16]:
    mem = memory_system.read(memory_ids[idx])
    print(f"--- Memory {idx+1} ---")
    print(f"  Content:  {mem.content[:80]}...")
    print(f"  Keywords: {mem.keywords}")
    print(f"  Context:  {mem.context[:100] if mem.context else 'N/A'}...")
    print(f"  Tags:     {mem.tags}")
    print(f"  Links:    {mem.links}")
    print()

# === Test search với nhiều query khác nhau ===
queries = [
    "container orchestration and deployment",
    "Python performance optimization",
    "cách cải thiện sức khỏe",
    "database and data storage",
    "machine learning và deep learning",
    "DevOps CI/CD best practices",
]

print(f"{'='*60}")
print("=== Search results ===\n")
for q in queries:
    print(f"Query: \"{q}\"")
    results = memory_system.search(q, k=3)
    for i, r in enumerate(results):
        print(f"  [{i+1}] {r['content'][:80]}...")
        print(f"      Tags: {r['tags']}")
    print()

# === Test agentic search ===
print(f"{'='*60}")
print("=== Agentic search ===\n")
agentic_results = memory_system.search_agentic("How to build scalable backend systems?", k=5)
print("Query: \"How to build scalable backend systems?\"")
for i, r in enumerate(agentic_results):
    print(f"  [{i+1}] {r['content'][:90]}...")
    print(f"      Tags: {r['tags']}")
print()

print("Done!")
