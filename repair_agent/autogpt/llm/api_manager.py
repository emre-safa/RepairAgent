from __future__ import annotations

from typing import List, Optional

import openai
from openai import Model

from autogpt.llm.base import CompletionModelInfo
from autogpt.logs import logger
from autogpt.singleton import Singleton


class ApiManager(metaclass=Singleton):
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0  # subset of total_completion_tokens
        self.total_cost = 0
        self.total_budget = 0
        self.models: Optional[list[Model]] = None
        # Per-model breakdown so different models (e.g. a reasoning smart_llm vs
        # a non-reasoning static_llm, each with its own pricing) can be reported
        # separately after a run. Keyed by model name -> running token/cost dict.
        self.per_model_stats: dict[str, dict] = {}
        # Per-call snapshot, populated on every API response so the cycle
        # summary can show reasoning vs visible breakdown against the cap.
        self.last_call_model: Optional[str] = None
        self.last_call_prompt_tokens: int = 0
        self.last_call_completion_tokens: int = 0
        self.last_call_reasoning_tokens: int = 0
        self.last_call_max_completion_tokens: Optional[int] = None

    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_cost = 0
        self.total_budget = 0.0
        self.models = None
        self.per_model_stats = {}
        self.last_call_model = None
        self.last_call_prompt_tokens = 0
        self.last_call_completion_tokens = 0
        self.last_call_reasoning_tokens = 0
        self.last_call_max_completion_tokens = None

    def set_last_call_cap(self, max_completion_tokens: Optional[int]) -> None:
        """Recorded by the request-side wrapper so the cycle summary can show
        how close the model got to its budget on the most recent call.
        """
        self.last_call_max_completion_tokens = max_completion_tokens

    def update_cost(self, prompt_tokens, completion_tokens, model, reasoning_tokens: int = 0):
        """
        Update the total cost, prompt tokens, and completion tokens.

        Args:
        prompt_tokens (int): The number of tokens used in the prompt.
        completion_tokens (int): The number of tokens used in the completion.
            For reasoning models this includes invisible reasoning tokens.
        model (str): The model used for the API call.
        reasoning_tokens (int): Subset of completion_tokens spent on invisible
            reasoning (0 for non-reasoning models).
        """
        # the .model property in API responses can contain version suffixes like -v2
        from autogpt.llm.providers.openai import ALL_MODELS

        model = model[:-3] if model.endswith("-v2") else model

        # Per-call snapshot — always recorded, regardless of cost-table coverage.
        self.last_call_model = model
        self.last_call_prompt_tokens = prompt_tokens
        self.last_call_completion_tokens = completion_tokens
        self.last_call_reasoning_tokens = reasoning_tokens

        # Dollar cost of this single call, priced from the per-model table.
        # Unknown models still have their tokens counted (so token totals stay
        # correct) but contribute $0 because we have no price for them.
        call_cost = 0.0
        if model in ALL_MODELS:
            model_info = ALL_MODELS[model]
            call_cost += prompt_tokens * model_info.prompt_token_cost / 1000
            if issubclass(type(model_info), CompletionModelInfo):
                call_cost += completion_tokens * model_info.completion_token_cost / 1000
        else:
            logger.warn(
                f"Unknown model '{model}' for cost tracking; "
                "tokens counted, cost recorded as $0."
            )

        # Global running totals.
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_reasoning_tokens += reasoning_tokens
        self.total_cost += call_cost

        # Per-model breakdown (mirrors the global totals but split by model).
        stats = self.per_model_stats.setdefault(
            model,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cost": 0.0,
            },
        )
        stats["calls"] += 1
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion_tokens
        stats["reasoning_tokens"] += reasoning_tokens
        stats["cost"] += call_cost

        logger.debug(f"Total running cost: ${self.total_cost:.3f}")

    def set_total_budget(self, total_budget):
        """
        Sets the total user-defined budget for API calls.

        Args:
        total_budget (float): The total budget for API calls.
        """
        self.total_budget = total_budget

    def get_total_prompt_tokens(self):
        """
        Get the total number of prompt tokens.

        Returns:
        int: The total number of prompt tokens.
        """
        return self.total_prompt_tokens

    def get_total_completion_tokens(self):
        """
        Get the total number of completion tokens.

        Returns:
        int: The total number of completion tokens.
        """
        return self.total_completion_tokens

    def get_total_reasoning_tokens(self):
        """
        Get the total number of invisible reasoning tokens (a subset of the
        completion tokens, produced only by reasoning models).

        Returns:
        int: The total number of reasoning tokens.
        """
        return self.total_reasoning_tokens

    def get_per_model_stats(self) -> dict:
        """
        Get the per-model token/cost breakdown accumulated so far.

        Returns a copy keyed by model name; each value carries
        calls / prompt_tokens / completion_tokens / reasoning_tokens / cost.
        For reasoning models, reasoning_tokens is the invisible subset of
        completion_tokens, so visible output = completion_tokens - reasoning_tokens.
        """
        return {model: dict(stats) for model, stats in self.per_model_stats.items()}

    def get_total_cost(self):
        """
        Get the total cost of API calls.

        Returns:
        float: The total cost of API calls.
        """
        return self.total_cost

    def get_total_budget(self):
        """
        Get the total user-defined budget for API calls.

        Returns:
        float: The total budget for API calls.
        """
        return self.total_budget

    def get_models(self, **openai_credentials) -> List[Model]:
        """
        Get list of available GPT models.

        Returns:
        list: List of available GPT models.

        """
        if self.models is None:
            all_models = openai.Model.list(**openai_credentials)["data"]
            self.models = [model for model in all_models if "gpt" in model["id"]]

        return self.models
