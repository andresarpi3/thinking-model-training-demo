#!/usr/bin/env python
"""Reward functions for GRPO training with optional consensus targets and log-scoring calibration."""

import re
import math
from collections import defaultdict
from enum import Enum

from tr_config import config
from dataset_utils import extract_confidence

# Toggle: when True, confidence targets are per-prompt consensus (fraction correct among gens)
#         when False, targets are per-sample R ∈ {0,1}
TARGET_CONSENSUS = True


class AnswerCorrectness(Enum):
    EXACT_MATCH = "exact_match"
    STRIPPED_MATCH = "stripped_match"
    CLOSE_NUMERICAL = "close_numerical"
    APPROXIMATE_NUMERICAL = "approximate_numerical"
    INCORRECT = "incorrect"
    INVALID = "invalid"


def classify_answer_correctness(extracted_answer, true_answer):
    """Classify the correctness of an extracted answer."""
    if extracted_answer is None:
        return AnswerCorrectness.INVALID

    if true_answer is None:
        return AnswerCorrectness.INVALID

    if extracted_answer == true_answer:
        return AnswerCorrectness.EXACT_MATCH

    if extracted_answer.strip() == true_answer.strip():
        return AnswerCorrectness.STRIPPED_MATCH

    try:
        extracted_num = float(extracted_answer.replace(",", "").strip())
        true_num = float(true_answer.replace(",", "").strip())
        ratio = extracted_num / true_num

        if 0.9 <= ratio <= 1.1:
            return AnswerCorrectness.CLOSE_NUMERICAL
        elif 0.8 <= ratio <= 1.2:
            return AnswerCorrectness.APPROXIMATE_NUMERICAL
        else:
            return AnswerCorrectness.INCORRECT
    except Exception:
        return AnswerCorrectness.INCORRECT


def create_reward_functions(use_confidence: bool = False):
    """Create reward functions with config-specific tokens."""
    prompts_cfg = config.prompts
    reasoning_end = prompts_cfg.reasoning_end
    solution_start = prompts_cfg.solution_start
    solution_end = prompts_cfg.solution_end
    confidence_start = prompts_cfg.confidence_start
    confidence_end = prompts_cfg.confidence_end

    solution_end_regex = rf"{re.escape(solution_end)}[\s]{{0,}}"

    def match_format_exactly(completions, **kwargs):
        if use_confidence:
            confidence_end_regex = rf"{re.escape(confidence_end)}[\s]{{0,}}"
            match_format = re.compile(
                rf"{re.escape(reasoning_end)}.*?{re.escape(solution_start)}(.+?){re.escape(solution_end)}.*?"
                rf"{re.escape(confidence_start)}(.+?){confidence_end_regex}[\s]{{0,}}$",
                flags=re.MULTILINE | re.DOTALL,
            )
        else:
            match_format = re.compile(
                rf"{re.escape(reasoning_end)}.*?{re.escape(solution_start)}(.+?){solution_end_regex}[\s]{{0,}}$",
                flags=re.MULTILINE | re.DOTALL,
            )

        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 3.0 if match_format.search(response) else 0.0
            scores.append(score)
        return scores

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            score = 0.0
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0

            if use_confidence:
                score += 0.5 if response.count(confidence_start) == 1 else -1.0
                score += 0.5 if response.count(confidence_end) == 1 else -1.0

            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        match_format = re.compile(
            rf"{re.escape(reasoning_end)}.*?{re.escape(solution_start)}(.+?){solution_end_regex}",
            flags=re.MULTILINE | re.DOTALL,
        )

        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            (m.group(1).strip() if (m := match_format.search(r)) else None) for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            correctness = classify_answer_correctness(guess, true_answer)
            if correctness == AnswerCorrectness.EXACT_MATCH:
                score = 5.0
            elif correctness == AnswerCorrectness.STRIPPED_MATCH:
                score = 3.5
            elif correctness == AnswerCorrectness.CLOSE_NUMERICAL:
                score = 2.0
            elif correctness == AnswerCorrectness.APPROXIMATE_NUMERICAL:
                score = 1.5
            elif correctness == AnswerCorrectness.INCORRECT:
                score = -2.5
            else:  # INVALID
                score = -4.5
            scores.append(score)
        return scores

    def confidence_calibration_reward(prompts, completions, answer, **kwargs):
        """
        Proper log scoring rule with optional per-prompt consensus targets.

        reward = p_target * log(c) + (1 - p_target) * log(1 - c)

        p_target:
            - If TARGET_CONSENSUS=False, p_target = R ∈ {0,1} (per-sample correctness)
            - If TARGET_CONSENSUS=True,  p_target = p̂ = fraction-correct within the same prompt group
        """
        eps = 1e-6
        SCALE = 5.0  # adjust to balance with other rewards

        # Precompile
        ans_rx = re.compile(
            rf"{re.escape(reasoning_end)}.*?{re.escape(solution_start)}(.+?){re.escape(solution_end)}",
            flags=re.MULTILINE | re.DOTALL,
        )

        N = len(completions)
        responses = [c[0]["content"] for c in completions]

        # 1) Per-sample correctness (used in both modes)
        is_correct = []
        for i in range(N):
            m = ans_rx.search(responses[i])
            extracted_answer = m.group(1).strip() if m else None
            corr = classify_answer_correctness(extracted_answer, answer[i])
            is_correct.append(corr in {AnswerCorrectness.EXACT_MATCH, AnswerCorrectness.STRIPPED_MATCH})

        # 2) Build per-index targets
        if TARGET_CONSENSUS:
            groups = defaultdict(list)
            for i, ptxt in enumerate(prompts):
                groups[str(ptxt)].append(i)

            p_target = [0.0] * N
            for idxs in groups.values():
                phat = sum(is_correct[j] for j in idxs) / max(1, len(idxs))
                for j in idxs:
                    p_target[j] = phat
        else:
            p_target = [1.0 if ok else 0.0 for ok in is_correct]
            
        print(f"Targets for this generation are: {p_target}")

        # 3) Score confidences with log rule
        scores = []
        for i in range(N):
            c = extract_confidence(responses[i], confidence_start, confidence_end)

            if c is None:
                scores.append(-80.0)  # missing confidence
                continue
            if not (0.0 <= c <= 1.0):
                scores.append(-80.0)  # out-of-bounds
                continue

            c = min(max(c, eps), 1.0 - eps)  # clip for stability
            pt = p_target[i]

            reward = pt * math.log(c) + (1.0 - pt) * math.log(1.0 - c)
            scores.append(SCALE * reward)

        return scores

    if use_confidence:
        return [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            confidence_calibration_reward,
        ], [0.0, 1.0, 0.0, 1.0]
    else:
        return [match_format_exactly, match_format_approximately, check_answer], None
