from dataclasses import dataclass

from agent.protocol import AgentAction, parse_action
from agent.search_tool import SearchTool


INVALID_ACTION_OBSERVATION = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between "
    "<search> and </search>. "
    "If I want to give the final answer, I should put the answer between "
    "<answer> and </answer>. "
    "Let me try again.\n"
)


@dataclass
class StepResult:
    action: AgentAction
    observation: str
    done: bool
    valid: bool
    is_search: bool


class SearchEnvironment:
    def __init__(
        self,
        search_tool: SearchTool,
    ):
        self.search_tool = search_tool

    def step(
        self,
        model_output: str,
    ) -> StepResult:
        action = parse_action(model_output)

        if action.type == "search":
            search_result = self.search_tool.run(
                action.content
            )

            observation = (
                "\n\n"
                "<information>"
                f"{search_result}"
                "</information>"
                "\n\n"
            )

            return StepResult(
                action=action,
                observation=observation,
                done=False,
                valid=True,
                is_search=True,
            )

        if action.type == "answer":
            return StepResult(
                action=action,
                observation="",
                done=True,
                valid=True,
                is_search=False,
            )

        return StepResult(
            action=action,
            observation=INVALID_ACTION_OBSERVATION,
            done=False,
            valid=False,
            is_search=False,
        )
    
    def batch_step(
        self,
        model_outputs: list[str],
    ) -> list[StepResult]:
        actions = [
            parse_action(output)
            for output in model_outputs
        ]

        search_indices = []
        search_queries = []

        for index, action in enumerate(actions):
            if action.type == "search":
                search_indices.append(index)
                search_queries.append(action.content)

        search_observations = {}

        if search_queries:
            search_results = self.search_tool.batch_run(
                search_queries
            )

            for index, search_result in zip(
                search_indices,
                search_results,
            ):
                search_observations[index] = search_result

        results = []

        for index, action in enumerate(actions):
            if action.type == "search":
                observation = (
                    "\n\n"
                    "<information>"
                    f"{search_observations[index]}"
                    "</information>"
                    "\n\n"
                )

                results.append(
                    StepResult(
                        action=action,
                        observation=observation,
                        done=False,
                        valid=True,
                        is_search=True,
                    )
                )

            elif action.type == "answer":
                results.append(
                    StepResult(
                        action=action,
                        observation="",
                        done=True,
                        valid=True,
                        is_search=False,
                    )
                )

            else:
                results.append(
                    StepResult(
                        action=action,
                        observation=INVALID_ACTION_OBSERVATION,
                        done=False,
                        valid=False,
                        is_search=False,
                    )
                )

        return results
