from dataclasses import dataclass
from typing import Literal, Optional, Protocol

from agent.environment import SearchEnvironment


TerminationReason = Literal["answer", "max_turns"]


class AgentModel(Protocol):
    def generate(self, context: str) -> str:
        ...


@dataclass
class AgentLoopResult:
    trajectory: str
    final_answer: Optional[str]
    done: bool
    turns: int
    search_count: int
    valid_action_count: int
    invalid_action_count: int
    termination_reason: TerminationReason


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
        initial_context: str,
    ) -> AgentLoopResult:
        trajectory = initial_context

        search_count = 0
        valid_action_count = 0
        invalid_action_count = 0

        for turn in range(1, self.max_turns + 1):
            model_output = self.model.generate(trajectory)

            trajectory += model_output

            step_result = self.environment.step(
                model_output
            )

            if step_result.valid:
                valid_action_count += 1
            else:
                invalid_action_count += 1

            if step_result.is_search:
                search_count += 1

            if step_result.done:
                final_answer = (
                    step_result.action.content
                    if step_result.action.type == "answer"
                    else None
                )

                return AgentLoopResult(
                    trajectory=trajectory,
                    final_answer=final_answer,
                    done=True,
                    turns=turn,
                    search_count=search_count,
                    valid_action_count=valid_action_count,
                    invalid_action_count=invalid_action_count,
                    termination_reason="answer",
                )

            trajectory += step_result.observation

        return AgentLoopResult(
            trajectory=trajectory,
            final_answer=None,
            done=False,
            turns=self.max_turns,
            search_count=search_count,
            valid_action_count=valid_action_count,
            invalid_action_count=invalid_action_count,
            termination_reason="max_turns",
        )