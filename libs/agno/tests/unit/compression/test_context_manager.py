"""Tests for unified CompressionManager."""

import pytest

from agno.compression import CompressionManager
from agno.models.message import Message


class TestCompressionManagerInit:
    """Test CompressionManager initialization."""

    def test_creates_both_managers_by_default(self):
        cm = CompressionManager()
        assert cm._tool_compressor is not None
        assert cm._history_compactor is not None

    def test_disables_tool_compression(self):
        cm = CompressionManager(compress_tool_results=False)
        assert cm._tool_compressor is None
        assert cm._history_compactor is not None

    def test_disables_history_compaction(self):
        cm = CompressionManager(compact_history=False)
        assert cm._tool_compressor is not None
        assert cm._history_compactor is None

    def test_disables_both(self):
        cm = CompressionManager(compress_tool_results=False, compact_history=False)
        assert cm._tool_compressor is None
        assert cm._history_compactor is None

    def test_backward_compat_properties(self):
        cm = CompressionManager()
        assert cm.compression_manager is cm._tool_compressor
        assert cm.context_compaction_manager is cm._history_compactor


class TestCompressionManagerShouldCompress:
    """Test should_compress_tools and should_compact_history."""

    def test_returns_false_when_below_limits(self):
        cm = CompressionManager(
            tool_results_limit=5,
            message_limit=10,
        )
        messages = [Message(role="user", content="hi")]
        assert cm.should_compress_tools(messages) is False
        assert cm.should_compact_history(messages) is False

    def test_returns_true_when_tool_limit_exceeded(self):
        cm = CompressionManager(tool_results_limit=2)
        messages = [
            Message(role="user", content="hi"),
            Message(role="tool", content="result1", tool_name="test"),
            Message(role="tool", content="result2", tool_name="test"),
        ]
        assert cm.should_compress_tools(messages) is True

    def test_returns_true_when_message_limit_exceeded(self):
        cm = CompressionManager(message_limit=3)
        messages = [Message(role="user", content=f"msg{i}") for i in range(5)]
        assert cm.should_compact_history(messages) is True


class TestCompressionManagerCompress:
    """Test compress method (unified escalation)."""

    def test_returns_original_when_no_compression_needed(self):
        cm = CompressionManager(
            tool_results_limit=10,
            message_limit=50,
        )
        messages = [Message(role="user", content="hi")]
        result = cm.compress(messages)
        assert result.compacted_messages == messages
        assert result.summary is None


@pytest.mark.asyncio
class TestCompressionManagerAsync:
    """Test async methods."""

    async def test_ashould_compress_tools_returns_false_when_below_limits(self):
        cm = CompressionManager(
            tool_results_limit=5,
            message_limit=10,
        )
        messages = [Message(role="user", content="hi")]
        assert await cm.ashould_compress_tools(messages) is False
        assert await cm.ashould_compact_history(messages) is False

    async def test_acompress_returns_original_when_no_compression_needed(self):
        cm = CompressionManager(
            tool_results_limit=10,
            message_limit=50,
        )
        messages = [Message(role="user", content="hi")]
        result = await cm.acompress(messages)
        assert result.compacted_messages == messages
        assert result.summary is None
