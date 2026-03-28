"""Multi-provider LLM client (Anthropic Claude + OpenAI) with retry and rate limiting."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Type, TypeVar

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.common.config import load_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class TokenUsage:
    """Track cumulative token usage across calls."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_calls = 0
        self.failed_calls = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary(self) -> str:
        return (
            f"Calls: {self.total_calls} (failed: {self.failed_calls}) | "
            f"Tokens: {self.total_tokens:,} (input: {self.input_tokens:,}, "
            f"output: {self.output_tokens:,})"
        )


class LLMClient:
    """Async LLM client supporting both Anthropic and OpenAI."""

    def __init__(self, model: str | None = None, concurrency: int | None = None):
        settings = load_settings()
        self.model = model or settings["teacher_model"]
        self.provider = settings.get("provider", "anthropic")
        self.semaphore = asyncio.Semaphore(concurrency or settings["rate_limit"]["concurrency"])
        self.usage = TokenUsage()

        if self.provider == "openai":
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=settings.get("openai_api_key"))
        else:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=settings.get("anthropic_api_key")
            )

    # Models that require max_completion_tokens instead of max_tokens and don't support temperature
    _REASONING_MODELS = {"gpt-5-mini", "gpt-5-nano", "gpt-5", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4"}

    def _openai_kwargs(self, temperature: float, max_tokens: int) -> dict:
        """Build OpenAI API kwargs, adapting for reasoning vs standard models."""
        is_reasoning = self.model in self._REASONING_MODELS
        kwargs = {"model": self.model}
        if is_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        return kwargs

    def _record_usage(self, usage):
        if self.provider == "openai":
            self.usage.input_tokens += usage.prompt_tokens
            self.usage.output_tokens += usage.completion_tokens
        else:
            self.usage.input_tokens += usage.input_tokens
            self.usage.output_tokens += usage.output_tokens
        self.usage.total_calls += 1

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry {retry_state.attempt_number} after error: {retry_state.outcome.exception()}"
        ),
    )
    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion."""
        async with self.semaphore:
            if self.provider == "openai":
                kwargs = self._openai_kwargs(temperature, max_tokens)
                kwargs["messages"] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                response = await self.client.chat.completions.create(**kwargs)
                self._record_usage(response.usage)
                return response.choices[0].message.content
            else:
                response = await self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._record_usage(response.usage)
                return response.content[0].text

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    async def complete_json(
        self,
        system: str,
        user: str,
        response_model: Type[T],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> T:
        """Generate a structured JSON response validated against a Pydantic model."""
        async with self.semaphore:
            if self.provider == "openai":
                kwargs = self._openai_kwargs(temperature, max_tokens)
                kwargs["messages"] = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                kwargs["response_format"] = {"type": "json_object"}
                response = await self.client.chat.completions.create(**kwargs)
                self._record_usage(response.usage)
                content = response.choices[0].message.content
            else:
                json_system = system + "\n\nYou MUST respond with valid JSON only. No other text."
                response = await self.client.messages.create(
                    model=self.model,
                    system=json_system,
                    messages=[
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": "{"},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._record_usage(response.usage)
                content = "{" + response.content[0].text

            data = json.loads(content)
            return response_model.model_validate(data)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    async def complete_multi_turn(
        self,
        messages: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a completion from a full message history."""
        async with self.semaphore:
            if self.provider == "openai":
                kwargs = self._openai_kwargs(temperature, max_tokens)
                kwargs["messages"] = messages
                response = await self.client.chat.completions.create(**kwargs)
                self._record_usage(response.usage)
                return response.choices[0].message.content
            else:
                system = ""
                conversation = []
                for msg in messages:
                    if msg["role"] == "system":
                        system = msg["content"]
                    else:
                        conversation.append({"role": msg["role"], "content": msg["content"]})

                response = await self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._record_usage(response.usage)
                return response.content[0].text

    async def complete_batch(
        self,
        tasks: list[dict],
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> list[str | None]:
        """Run multiple completions concurrently."""

        async def _run(task: dict) -> str | None:
            try:
                return await self.complete(
                    system=task["system"],
                    user=task["user"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                self.usage.failed_calls += 1
                logger.error(f"Batch call failed: {e}")
                return None

        return await asyncio.gather(*[_run(t) for t in tasks])
