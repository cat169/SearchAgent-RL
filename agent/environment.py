from dataclasses import dataclass

from agent.protocol import AgentAction, parse_action
from agent.search_tool import SearchTool


@dataclass
class StepResult:
    action: AgentAction
    observation: str

    @property
    def valid(self) -> bool:
        return self.action.type != "invalid"

    @property
    def done(self) -> bool:
        return self.action.type == "answer"

    @property
    def is_search(self) -> bool:
        return self.action.type == "search"


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
            )

        if action.type == "answer":
            return StepResult(
                action=action,
                observation="",
            )

        return StepResult(
            action=action,
            observation="",
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
                    )
                )

            elif action.type == "answer":
                results.append(
                    StepResult(
                        action=action,
                        observation="",
                    )
                )

            else:
                results.append(
                    StepResult(
                        action=action,
                        observation="",
                    )
                )

        return results
