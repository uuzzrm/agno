"""Regression tests for followups after continuing a paused team run."""

import json
from typing import Any, AsyncIterator, Iterator, List

import pytest

from agno.models.base import Model
from agno.models.response import ModelResponse, ModelResponseEvent
from agno.run.team import TeamRunEvent
from agno.team import Team
from agno.tools import tool


class _ScriptedModel(Model):
    """Return a fixed sequence of responses without making a provider call."""

    def __init__(self, script: List[tuple]):
        super().__init__(id="scripted", name="scripted", provider="test")
        self._script = script
        self._index = 0

    def _next(self) -> ModelResponse:
        turn = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1

        if turn[0] == "tool":
            _, name, args, call_id = turn
            return ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            )

        response = ModelResponse(content=turn[1], role="assistant")
        response.event = ModelResponseEvent.assistant_response.value
        return response

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._next()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._next()

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        yield self._next()

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


@tool(requires_confirmation=True)
def approve_deployment(environment: str) -> str:
    return f"Deployment approved for {environment}"


def _team() -> Team:
    return Team(
        model=_ScriptedModel(
            [
                ("tool", "approve_deployment", {"environment": "staging"}, "call-1"),
                ("content", "The deployment is ready."),
                ("content", '{"suggestions": ["Review the deployment logs"]}'),
            ]
        ),
        members=[],
        tools=[approve_deployment],
        followups=True,
        num_followups=1,
        telemetry=False,
    )


@pytest.mark.asyncio
async def test_async_continue_generates_followups_after_confirmation():
    team = _team()

    paused = await team.arun("Deploy the app", session_id="continue-followups")
    assert paused.is_paused

    paused.active_requirements[0].confirm()
    result = await team.acontinue_run(paused)

    assert result.content == "The deployment is ready."
    assert result.followups == ["Review the deployment logs"]


def test_sync_continue_generates_followups_after_confirmation():
    team = _team()

    paused = team.run("Deploy the app", session_id="continue-followups")
    assert paused.is_paused

    paused.active_requirements[0].confirm()
    result = team.continue_run(paused)

    assert result.content == "The deployment is ready."
    assert result.followups == ["Review the deployment logs"]


def test_sync_stream_continue_emits_followups_after_confirmation():
    team = _team()

    paused = team.run("Deploy the app", session_id="continue-followups")
    assert paused.is_paused

    paused.active_requirements[0].confirm()
    events = list(team.continue_run(paused, stream=True, stream_events=True))
    followups_events = [event for event in events if event.event == TeamRunEvent.followups_completed.value]

    assert len(followups_events) == 1
    assert followups_events[0].followups == ["Review the deployment logs"]


@pytest.mark.asyncio
async def test_async_stream_continue_emits_followups_after_confirmation():
    team = _team()

    paused = await team.arun("Deploy the app", session_id="continue-followups")
    assert paused.is_paused

    paused.active_requirements[0].confirm()
    events = [event async for event in team.acontinue_run(paused, stream=True, stream_events=True)]
    followups_events = [event for event in events if event.event == TeamRunEvent.followups_completed.value]

    assert len(followups_events) == 1
    assert followups_events[0].followups == ["Review the deployment logs"]
