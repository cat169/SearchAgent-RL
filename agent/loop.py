from dataclasses import dataclass
from typing import Literal, Optional, Protocol

from agent.environment import SearchEnvironment
from agent.protocol import AgentAction


TerminationReason = Literal["answer", "invalid_action", "max_turns"]


class AgentModel(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        ...


@dataclass
class AgentStep:
    model_output: str
    action: AgentAction
    observation: str


@dataclass
class AgentLoopResult:
    steps: list[AgentStep]
    termination_reason: TerminationReason

    @property
    def turns(self) -> int:
        return len(self.steps)

    @property
    def search_count(self) -> int:
        return sum(step.action.type == "search" for step in self.steps)

    @property
    def done(self) -> bool:
        return self.termination_reason == "answer"

    @property
    def final_answer(self) -> Optional[str]:
        if self.termination_reason == "answer":
            return self.steps[-1].action.content
        return None


class AgentLoop:
    def __init__(
        self,
        model: AgentModel,
        environment: SearchEnvironment,
        max_turns: int = 5,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")

        self.model = model
        self.environment = environment
        self.max_turns = max_turns

    def run(
        self,
        initial_user_content: str,
    ) -> AgentLoopResult:
        messages = [
            {
                "role": "user",
                "content": initial_user_content,
            }
        ]
        steps = []

        for _ in range(self.max_turns):
            model_output = self.model.generate(messages)
            messages.append(
                {
                    "role": "assistant",
                    "content": model_output,
                }
            )
            step_result = self.environment.step(
                model_output
            )
            steps.append(
                AgentStep(
                    model_output=model_output,
                    action=step_result.action,
                    observation=step_result.observation,
                )
            )

            if step_result.action.type == "search":
                messages.append(
                    {
                        "role": "user",
                        "content": step_result.observation,
                    }
                )
                continue

            if step_result.action.type == "answer":
                return AgentLoopResult(
                    steps=steps,
                    termination_reason="answer",
                )

            return AgentLoopResult(
                steps=steps,
                termination_reason="invalid_action",
            )

        return AgentLoopResult(
            steps=steps,
            termination_reason="max_turns",
        )
