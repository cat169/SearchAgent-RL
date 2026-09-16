from typing import Protocol


class BatchAgentModel(Protocol):
    def generate_batch(
        self,
        contexts: list[str],
    ) -> list[str]:
        ...

from dataclasses import dataclass
from typing import Optional

from agent.environment import SearchEnvironment


@dataclass
class BatchTrajectoryResult:
    trajectory: str
    final_answer: Optional[str]
    done: bool
    turns: int
    search_count: int
    valid_action_count: int
    invalid_action_count: int
    termination_reason: str


class BatchAgentLoop:
    def __init__(
        self,
        model,
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
        initial_contexts: list[str],
    ) -> list[BatchTrajectoryResult]:

        batch_size = len(initial_contexts)

        trajectories = list(initial_contexts)

        active = [True] * batch_size

        final_answers = [None] * batch_size
        turns = [0] * batch_size
        search_counts = [0] * batch_size
        valid_counts = [0] * batch_size
        invalid_counts = [0] * batch_size

        for _ in range(self.max_turns):
            active_indices = [
                index
                for index, is_active in enumerate(active)
                if is_active
            ]

            if not active_indices:
                break

            active_contexts = [
                trajectories[index]
                for index in active_indices
            ]

            model_outputs = self.model.generate_batch(
                active_contexts
            )

            if len(model_outputs) != len(active_indices):
                raise ValueError(
                    "model output batch size does not match "
                    "active trajectory count"
                )

            step_results = self.environment.batch_step(
                model_outputs
            )

            if len(step_results) != len(active_indices):
                raise ValueError(
                    "environment result batch size does not match "
                    "active trajectory count"
                )

            for local_index, global_index in enumerate(
                active_indices
            ):
                model_output = model_outputs[local_index]
                step_result = step_results[local_index]

                trajectories[global_index] += model_output
                turns[global_index] += 1

                if step_result.valid:
                    valid_counts[global_index] += 1
                else:
                    invalid_counts[global_index] += 1

                if step_result.is_search:
                    search_counts[global_index] += 1

                if step_result.done:
                    active[global_index] = False

                    if step_result.action.type == "answer":
                        final_answers[global_index] = (
                            step_result.action.content
                        )

                else:
                    trajectories[global_index] += (
                        step_result.observation
                    )

        results = []

        for index in range(batch_size):
            done = not active[index]

            results.append(
                BatchTrajectoryResult(
                    trajectory=trajectories[index],
                    final_answer=final_answers[index],
                    done=done,
                    turns=turns[index],
                    search_count=search_counts[index],
                    valid_action_count=valid_counts[index],
                    invalid_action_count=invalid_counts[index],
                    termination_reason=(
                        "answer"
                        if done
                        else "max_turns"
                    ),
                )
            )

        return results