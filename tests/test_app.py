import unittest
from llama_index.core.llms import ChatMessage, MessageRole

# Adjust the import path based on your project structure
# Assuming 'app.py' is in the parent directory of 'tests/'
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import format_chat_history # Or wherever this function is

class TestAppFormatChatHistory(unittest.TestCase):

    def test_empty_history(self):
        streamlit_messages = []
        expected_llama_messages = []
        self.assertEqual(format_chat_history(streamlit_messages), expected_llama_messages)

    def test_user_message(self):
        streamlit_messages = [{"role": "user", "content": "Hello"}]
        expected_llama_messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        self.assertEqual(format_chat_history(streamlit_messages), expected_llama_messages)

    def test_assistant_message(self):
        streamlit_messages = [{"role": "assistant", "content": "Hi there"}]
        expected_llama_messages = [ChatMessage(role=MessageRole.ASSISTANT, content="Hi there")]
        self.assertEqual(format_chat_history(streamlit_messages), expected_llama_messages)

    def test_mixed_messages(self):
        streamlit_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        expected_llama_messages = [
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there"),
            ChatMessage(role=MessageRole.USER, content="How are you?")
        ]
        self.assertEqual(format_chat_history(streamlit_messages), expected_llama_messages)

    def test_message_with_additional_keys(self):
        # Ensure only 'role' and 'content' are used for ChatMessage
        streamlit_messages = [{"role": "user", "content": "Test", "other_key": "value"}]
        expected_llama_messages = [ChatMessage(role=MessageRole.USER, content="Test")]
        # Need to compare relevant fields if ChatMessage objects don't ignore extra attrs on init
        # For LlamaIndex ChatMessage, direct equality should work if it only considers role and content.
        result = format_chat_history(streamlit_messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, MessageRole.USER)
        self.assertEqual(result[0].content, "Test")


if __name__ == '__main__':
    unittest.main()
