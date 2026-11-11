import unittest

from evaluation.evaluate_agent import exact_match
from training.grpo.reward import compute_score


class ExactMatchRewardTests(unittest.TestCase):
    def score(self, prediction, golds, termination="answer", strict=True, response=""):
        return compute_score(
            data_source="nq",
            solution_str=response,
            ground_truth=golds,
            extra_info={
                "final_answer": prediction,
                "termination_reason": termination,
                "search_count": 1,
                "strict_protocol_valid": strict,
            },
        )

    def test_valid_correct_answer(self):
        score = self.score("Paris", ["Paris"])
        self.assertEqual(score, 1.0)
        self.assertIsInstance(score, float)

    def test_valid_incorrect_answer(self):
        self.assertEqual(self.score("London", ["Paris"]), 0.0)

    def test_matches_nonfirst_gold_answer(self):
        self.assertEqual(self.score("NYC", ["New York City", "NYC"]), 1.0)

    def test_normalization_matches_evaluation(self):
        for prediction, gold in [
            ("  THE   United States! ", "United States"),
            ("An APPLE.", "apple"),
            ("Paris", "London"),
        ]:
            with self.subTest(prediction=prediction, gold=gold):
                self.assertEqual(self.score(prediction, [gold]), exact_match(prediction, gold))

    def test_invalid_action(self):
        self.assertEqual(self.score("Paris", ["Paris"], termination="invalid_action"), 0.0)

    def test_max_turns(self):
        self.assertEqual(self.score("Paris", ["Paris"], termination="max_turns"), 0.0)

    def test_length_limit(self):
        self.assertEqual(self.score("Paris", ["Paris"], termination="length_limit"), 0.0)

    def test_answer_with_invalid_protocol(self):
        self.assertEqual(self.score("Paris", ["Paris"], strict=False), 0.0)

    def test_uses_final_answer_instead_of_response_text(self):
        response = "<information>Paris</information><answer>Paris</answer>"
        self.assertEqual(self.score("London", ["Paris"], response=response), 0.0)
        self.assertEqual(self.score("Paris", ["Paris"], response="unrelated response"), 1.0)


if __name__ == "__main__":
    unittest.main()
