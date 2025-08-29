import asyncio
import httpx
import pytest
import uvicorn
from threading import Thread
import time
import json
from server.main import app
import os
from langchain_core.tracers.log_stream import RunLogPatch

# Set dummy API keys for testing
os.environ["GOOGLE_API_KEY"] = "test"
os.environ["TAVILY_API_KEY"] = "test"
os.environ["OPENROUTER_API_KEY"] = "test"
os.environ["MISTRAL_API_KEY"] = "test"

class ServerThread(Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.server = None

    def run(self):
        config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
        self.server = uvicorn.Server(config)
        self.server.run()

    def shutdown(self):
        if self.server:
            self.server.should_exit = True

@pytest.fixture(scope="module")
def server():
    thread = ServerThread()
    thread.start()
    time.sleep(5)  # Give the server time to start
    yield
    thread.shutdown()
    thread.join()

class MockAgent:
    def __init__(self, stream):
        self.stream = stream

    async def astream_log(self, *args, **kwargs):
        for item in self.stream:
            yield item
            await asyncio.sleep(0.01)

@pytest.mark.asyncio
async def test_simple_streaming(server, mocker):
    """Test streaming for a simple query that doesn't require a tool."""

    # Mock the agent to return a predefined stream
    stream = [
        RunLogPatch([{'op': 'add', 'path': '/streamed_output/-', 'value': {'content': 'Hello'}}]),
        RunLogPatch([{'op': 'add', 'path': '/streamed_output/-', 'value': {'content': ', world!'}}]),
    ]
    mock_agent = MockAgent(stream)
    mocker.patch('server.main.create_agent', return_value=mock_agent)

    url = "http://127.0.0.1:8000/chat/stream"
    messages = [{"role": "user", "content": "Hello, world!"}]
    data = {"messages": json.dumps(messages)}

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, data=data, timeout=30) as response:
            assert response.status_code == 200

            all_content = ""
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    try:
                        content = json.loads(line[5:])
                        all_content += str(content)
                    except json.JSONDecodeError:
                        continue

            assert "'type': 'delta', 'text': 'Hello'" in all_content
            assert "'type': 'delta', 'text': ', world!'" in all_content
            assert '"type": "done"' in all_content

@pytest.mark.asyncio
async def test_tool_use_streaming(server, mocker):
    """Test streaming for a query that should trigger a tool."""

    # Mock the agent to return a predefined stream with a tool call
    stream = [
        RunLogPatch([{'op': 'add', 'path': '/logs/ChatOpenAI/tool_calls/-', 'value': {'name': 'tavily_search', 'args': {'query': 'capital of France'}, 'id': 'call_123'}}]),
        RunLogPatch([{'op': 'add', 'path': '/streamed_output/-', 'value': {'content': 'The capital of France is Paris.'}}]),
    ]
    mock_agent = MockAgent(stream)
    mocker.patch('server.main.create_agent', return_value=mock_agent)

    url = "http://127.0.0.1:8000/chat/stream"
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    data = {"messages": json.dumps(messages)}

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, data=data, timeout=30) as response:
            assert response.status_code == 200

            all_content = ""
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    try:
                        content = json.loads(line[5:])
                        all_content += str(content)
                    except json.JSONDecodeError:
                        continue

            assert "'type': 'status', 'message': 'Searching the web...'" in all_content
            assert "'type': 'delta', 'text': 'The capital of France is Paris.'" in all_content
            assert '"type": "done"' in all_content
