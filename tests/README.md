# Test Architecture

This test suite is split by scope so failures point to the right layer quickly:

- `tests/unit/`: pure behavior and adapter tests with full mocking.
- `tests/integration/`: real `AgenticMemorySystem` flows using deterministic in-memory doubles for the LLM and vector store.
- `tests/stress/`: bulk lifecycle coverage to catch consistency regressions under sustained add, update, link, delete, and rebuild activity.

The suite is intentionally deterministic:

- no network calls
- no API keys
- no real Ollama, OpenAI, or SGLang servers
- no sentence-transformer downloads during test execution

Core test doubles live in [`helpers.py`](/home/tcuong1000/os-twin/A-mem-sys/tests/helpers.py).
