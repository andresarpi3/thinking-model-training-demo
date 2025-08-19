#!/usr/bin/env python
"""Reward functions for GRPO training."""

import re


def create_reward_functions(config):
    """Create reward functions with config-specific tokens"""
    prompts = config["prompts"]
    reasoning_end = prompts["reasoning_end"]
    solution_start = prompts["solution_start"]
    solution_end = prompts["solution_end"]
    
    def match_format_exactly(completions, **kwargs):
        solution_end_regex = r"</SOLUTION>[\s]{0,}"
        match_format = re.compile(
            rf"{re.escape(reasoning_end)}.*?{re.escape(solution_start)}(.+?){solution_end_regex}[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
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
            score = 0
            score += 0.5 if response.count(reasoning_end) == 1 else -1.0
            score += 0.5 if response.count(solution_start) == 1 else -1.0
            score += 0.5 if response.count(solution_end) == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(prompts, completions, answer, **kwargs):
        solution_end_regex = r"</SOLUTION>[\s]{0,}"
        match_format = re.compile(
            rf"{re.escape(reasoning_end)}.*?{re.escape(solution_start)}(.+?){solution_end_regex}",
            flags=re.MULTILINE | re.DOTALL
        )

        responses = [completion[0]["content"] for completion in completions]
        extracted_responses = [
            match.group(1).strip() if (match := match_format.search(r)) else None
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.0)
                continue

            if guess == true_answer:
                score = 5.0
            elif guess.strip() == true_answer.strip():
                score = 3.5
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        score = 2.0
                    elif 0.8 <= ratio <= 1.2:
                        score = 1.5
                    else:
                        score = -2.5
                except:
                    score = -4.5
            scores.append(score)
        return scores
    
    return [match_format_exactly, match_format_approximately, check_answer]