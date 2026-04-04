# Backend Test Notes

SGLang behavior is covered in [`test_llm_controller.py`](/home/tcuong1000/os-twin/A-mem-sys/tests/unit/test_llm_controller.py):

- request payload construction
- JSON schema stringification
- fallback behavior on transport or server failure

These are unit tests only. They mock HTTP requests and do not require a running SGLang server.
