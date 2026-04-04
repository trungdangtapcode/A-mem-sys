import json
import unittest
from unittest.mock import Mock, patch

from agentic_memory.llm_controller import (
    BaseLLMController,
    LLMController,
    OllamaController,
    OpenAIController,
    SGLangController,
)


class DummyLLMController(BaseLLMController):
    def get_completion(self, prompt: str) -> str:
        del prompt
        return "{}"


class TestBaseLLMController(unittest.TestCase):
    def setUp(self):
        self.controller = DummyLLMController()

    def test_generate_empty_response_matches_schema(self):
        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {
                        "keywords": {"type": "array"},
                        "context": {"type": "string"},
                        "score": {"type": "number"},
                        "enabled": {"type": "boolean"},
                    }
                }
            }
        }

        self.assertEqual(
            self.controller._generate_empty_response(response_format),
            {
                "keywords": [],
                "context": "",
                "score": 0,
                "enabled": False,
            },
        )


class TestSGLangController(unittest.TestCase):
    def setUp(self):
        self.controller = SGLangController(
            model="meta-llama/Llama-3.1-8B-Instruct",
            sglang_host="http://localhost",
            sglang_port=30000,
        )

    @patch("agentic_memory.llm_controller.requests.post")
    def test_get_completion_success_uses_stringified_schema(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"text": '{"keywords": ["memory"]}'}
        mock_post.return_value = mock_response

        response_format = {
            "json_schema": {
                "schema": {
                    "properties": {"keywords": {"type": "array"}},
                }
            }
        }

        result = self.controller.get_completion("test prompt", response_format, temperature=0.4)

        self.assertEqual(result, '{"keywords": ["memory"]}')
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["text"], "test prompt")
        self.assertEqual(payload["sampling_params"]["temperature"], 0.4)
        self.assertIsInstance(payload["sampling_params"]["json_schema"], str)
        self.assertEqual(
            json.loads(payload["sampling_params"]["json_schema"]),
            response_format["json_schema"]["schema"],
        )

    @patch("agentic_memory.llm_controller.requests.post")
    def test_get_completion_returns_empty_payload_on_server_error(self, mock_post):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        result = self.controller.get_completion(
            "test prompt",
            {"json_schema": {"schema": {"properties": {"context": {"type": "string"}}}}},
        )

        self.assertEqual(json.loads(result), {"context": ""})


class TestLLMControllerDispatch(unittest.TestCase):
    def test_openai_backend_selection_is_patchable(self):
        with patch.object(OpenAIController, "__init__", return_value=None):
            controller = LLMController(backend="openai", model="gpt-4o-mini", api_key="test-key")

        self.assertIsInstance(controller.llm, OpenAIController)

    def test_ollama_backend_selection_is_patchable(self):
        with patch.object(OllamaController, "__init__", return_value=None):
            controller = LLMController(backend="ollama", model="llama2")

        self.assertIsInstance(controller.llm, OllamaController)

    def test_invalid_backend_raises(self):
        with self.assertRaises(ValueError):
            LLMController(backend="invalid")
