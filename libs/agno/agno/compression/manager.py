from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from agno.compression._history_compaction import CompactionResult, ContextCompactionManager
from agno.compression._tool_compression import CompressionManager as ToolCompressionManager
from agno.models.base import Model
from agno.models.message import Message
from agno.utils.log import log_debug

if TYPE_CHECKING:
    from agno.metrics import RunMetrics
    from agno.run.agent import RunOutput


@dataclass
class CompressionManager:
    """Unified context manager for tool compression and history compaction.

    Single entry point for all context management. Internally creates and manages:
    - Tool compression (lightweight, compresses individual tool results)
    - History compaction (heavier, summarizes old conversation history)

    Usage:
        context_manager = ContextManager(
            model=OpenAI(id="gpt-4o-mini"),
            compress_tool_results=True,
            compact_history=True,
        )
    """

    model: Optional[Model] = None

    # Tool compression settings
    compress_tool_results: bool = True
    tool_results_limit: Optional[int] = None  # trigger after N tool results
    tool_token_limit: Optional[int] = None  # trigger at N tokens
    tool_compression_instructions: Optional[str] = None

    # History compaction settings
    compact_history: bool = True
    message_limit: Optional[int] = None  # trigger at N messages (default: 10)
    history_token_limit: Optional[int] = None  # trigger at N tokens
    keep_recent: int = 10  # messages to keep intact
    preserve_user_budget: int = 20_000  # token budget for preserving user messages
    compaction_instructions: Optional[str] = None

    stats: Dict[str, Any] = field(default_factory=dict)

    # Internal managers
    _tool_compressor: Optional[ToolCompressionManager] = field(default=None, repr=False)
    _history_compactor: Optional[ContextCompactionManager] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.compress_tool_results:
            self._tool_compressor = ToolCompressionManager(
                model=self.model,
                compress_tool_results=True,
                compress_tool_results_limit=self.tool_results_limit,
                compress_token_limit=self.tool_token_limit,
                compress_tool_call_instructions=self.tool_compression_instructions,
            )

        if self.compact_history:
            self._history_compactor = ContextCompactionManager(
                model=self.model,
                message_limit=self.message_limit,
                token_limit=self.history_token_limit,
                keep_recent=self.keep_recent,
                preserve_user_budget=self.preserve_user_budget,
                instructions=self.compaction_instructions,
            )

    # --- Backward compatibility properties ---

    @property
    def compression_manager(self) -> Optional[ToolCompressionManager]:
        """Access internal tool compressor (for _run.py call sites)."""
        return self._tool_compressor

    @property
    def context_compaction_manager(self) -> Optional[ContextCompactionManager]:
        """Access internal history compactor (for _run.py call sites)."""
        return self._history_compactor

    # --- Tool compression (mid-loop) ---

    def should_compress_tools(
        self,
        messages: List[Message],
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """Check if tool results should be compressed."""
        if self._tool_compressor is None:
            return False
        return self._tool_compressor.should_compress(messages, tools, model, response_format)

    async def ashould_compress_tools(
        self,
        messages: List[Message],
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> bool:
        """Async check if tool results should be compressed."""
        if self._tool_compressor is None:
            return False
        return await self._tool_compressor.ashould_compress(messages, tools, model, response_format)

    def compress_tools(
        self,
        messages: List[Message],
        run_metrics: Optional["RunMetrics"] = None,
    ) -> None:
        """Compress tool results in-place (sets msg.compressed_content)."""
        if self._tool_compressor is None:
            return
        log_debug("[CONTEXT_MANAGER] compress_tools()")
        self._tool_compressor.compress(messages, run_metrics)
        self._sync_stats()

    async def acompress_tools(
        self,
        messages: List[Message],
        run_metrics: Optional["RunMetrics"] = None,
    ) -> None:
        """Async compress tool results in-place."""
        if self._tool_compressor is None:
            return
        log_debug("[CONTEXT_MANAGER] acompress_tools()")
        await self._tool_compressor.acompress(messages, run_metrics)
        self._sync_stats()

    # --- History compaction (pre-loop) ---

    def should_compact_history(self, messages: List[Message]) -> bool:
        """Check if history should be compacted."""
        if self._history_compactor is None:
            return False
        return self._history_compactor.should_compact(messages)

    async def ashould_compact_history(self, messages: List[Message]) -> bool:
        """Async check if history should be compacted."""
        if self._history_compactor is None:
            return False
        return await self._history_compactor.ashould_compact(messages)

    def compact_history(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
    ) -> CompactionResult:
        """Compact history into summary message."""
        if self._history_compactor is None:
            return CompactionResult(compacted_messages=messages, summary=None)
        log_debug("[CONTEXT_MANAGER] compact_history()")
        result = self._history_compactor.compact(messages, run_response, run_metrics)
        self._sync_stats()
        return result

    async def acompact_history(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
    ) -> CompactionResult:
        """Async compact history into summary message."""
        if self._history_compactor is None:
            return CompactionResult(compacted_messages=messages, summary=None)
        log_debug("[CONTEXT_MANAGER] acompact_history()")
        result = await self._history_compactor.acompact(messages, run_response, run_metrics)
        self._sync_stats()
        return result

    # --- Unified compress (cheapest-first escalation) ---

    def compress(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> CompactionResult:
        """Compress context using cheapest-first escalation.

        1. Tool compression first (lightweight)
        2. History compaction if still needed (heavier)
        """
        # 1. Tool compression
        if self.should_compress_tools(messages, tools, model, response_format):
            self.compress_tools(messages, run_metrics)

        # 2. History compaction
        if self.should_compact_history(messages):
            return self.compact_history(messages, run_response, run_metrics)

        return CompactionResult(compacted_messages=messages, summary=None)

    async def acompress(
        self,
        messages: List[Message],
        run_response: Optional["RunOutput"] = None,
        run_metrics: Optional["RunMetrics"] = None,
        tools: Optional[List] = None,
        model: Optional[Model] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
    ) -> CompactionResult:
        """Async compress context using cheapest-first escalation."""
        # 1. Tool compression
        if await self.ashould_compress_tools(messages, tools, model, response_format):
            await self.acompress_tools(messages, run_metrics)

        # 2. History compaction
        if await self.ashould_compact_history(messages):
            return await self.acompact_history(messages, run_response, run_metrics)

        return CompactionResult(compacted_messages=messages, summary=None)

    def _sync_stats(self) -> None:
        """Aggregate stats from internal managers."""
        if self._tool_compressor is not None:
            for key, value in self._tool_compressor.stats.items():
                self.stats[f"tool_{key}"] = value
        if self._history_compactor is not None:
            for key, value in self._history_compactor.stats.items():
                self.stats[f"history_{key}"] = value
