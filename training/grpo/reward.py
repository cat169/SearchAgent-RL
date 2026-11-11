"""Terminal Exact Match reward for the veRL search Agent."""

from evaluation.evaluate_agent import exact_match


def compute_score(data_source, solution_str, ground_truth, extra_info):
    if (
        extra_info["termination_reason"] != "answer"
        or not extra_info["strict_protocol_valid"]
    ):
        return 0.0

    prediction = extra_info["final_answer"]
    return float(any(exact_match(prediction, answer) for answer in ground_truth))
