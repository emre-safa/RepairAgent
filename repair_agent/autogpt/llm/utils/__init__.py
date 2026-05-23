from __future__ import annotations

from typing import List, Literal, Optional

import openai
from colorama import Fore

from autogpt.config import Config

from ..api_manager import ApiManager
from ..base import (
    ChatModelResponse,
    ChatSequence,
    FunctionCallDict,
    Message,
    ResponseMessageDict,
)
from ..providers import openai as iopenai
from ..providers import anthropic as ianthropic
from ..providers.openai import (
    ALL_CHAT_MODELS,
    OpenAIFunctionCall,
    OpenAIFunctionSpec,
    count_openai_functions_tokens,
)
from ..providers.anthropic import is_anthropic_model
from .token_counter import *

# Models that require max_completion_tokens instead of max_tokens (and reject
# temperature). Seeded with known prefixes so the first call doesn't waste a
# round-trip discovering this; augmented at runtime for any unknown model that
# surfaces the same incompatibility.
_REASONING_MODELS: set[str] = set()
_REASONING_MODEL_PREFIXES: tuple[str, ...] = ("o1", "o3", "o4", "gpt-5")

# Reasoning models that also reject response_format={"type":"json_object"}.
# Discovered at runtime via a retry; gpt-5 series supports it, o1 series does not.
_NO_RESPONSE_FORMAT_MODELS: set[str] = set()


def _is_reasoning_model(model: str) -> bool:
    return model in _REASONING_MODELS or model.startswith(_REASONING_MODEL_PREFIXES)


def _supports_response_format(model: str) -> bool:
    if is_anthropic_model(model):
        return False
    return model not in _NO_RESPONSE_FORMAT_MODELS


def call_ai_function(
    function: str,
    args: list,
    description: str,
    config: Config,
    model: Optional[str] = None,
) -> str:
    """Call an AI function

    This is a magic function that can do anything with no-code. See
    https://github.com/Torantulino/AI-Functions for more info.

    Args:
        function (str): The function to call
        args (list): The arguments to pass to the function
        description (str): The description of the function
        model (str, optional): The model to use. Defaults to None.

    Returns:
        str: The response from the function
    """
    if model is None:
        model = config.smart_llm
    # For each arg, if any are None, convert to "None":
    args = [str(arg) if arg is not None else "None" for arg in args]
    # parse args to comma separated string
    arg_str: str = ", ".join(args)

    prompt = ChatSequence.for_model(
        model,
        [
            Message(
                "system",
                f"You are now the following python function: ```# {description}"
                f"\n{function}```\n\nOnly respond with your `return` value.",
            ),
            Message("user", arg_str),
        ],
    )
    return create_chat_completion(prompt=prompt, temperature=0, config=config).content


def create_text_completion(
    prompt: str,
    config: Config,
    model: Optional[str],
    temperature: Optional[float],
    max_output_tokens: Optional[int],
) -> str:
    if model is None:
        model = config.fast_llm
    if temperature is None:
        temperature = config.temperature

    kwargs = {"model": model}
    kwargs.update(config.get_openai_credentials(model))

    response = iopenai.create_text_completion(
        prompt=prompt,
        **kwargs,
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    logger.debug(f"Response: {response}")

    return response.choices[0].text


# Overly simple abstraction until we create something better
def create_chat_completion(
    prompt: ChatSequence,
    config: Config,
    functions: Optional[List[OpenAIFunctionSpec]] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatModelResponse:
    """Create a chat completion using the OpenAI or Anthropic API

    Args:
        messages (List[Message]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.

    Returns:
        str: The response from the chat completion
    """

    if model is None:
        model = prompt.model.name
    if temperature is None:
        temperature = config.temperature
    if max_tokens is None:
        prompt_tlength = prompt.token_length
        model_max = ALL_CHAT_MODELS[model].max_tokens if model in ALL_CHAT_MODELS else 128000
        max_tokens = (
            min(model_max - prompt_tlength - 1, 4000)
        )  # the -1 is just here because we have a bug and we don't know how to fix it. When using gpt-4-0314 we get a token error.
        logger.debug(f"Prompt length: {prompt_tlength} tokens")
        if functions and not is_anthropic_model(model):
            functions_tlength = count_openai_functions_tokens(functions, model)
            max_tokens -= functions_tlength
            logger.debug(f"Functions take up {functions_tlength} tokens in API call")

    logger.debug(
        f"{Fore.GREEN}Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}{Fore.RESET}"
    )
    with open("model_logging_temp.txt", "w") as mlt:
        mlt.write(f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")

    chat_completion_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # Anthropic models don't support response_format or OpenAI functions.
    # Some OpenAI reasoning models (e.g. o1 series) also reject it.
    if _supports_response_format(model):
        chat_completion_kwargs["response_format"] = { "type": "json_object" }

    for plugin in config.plugins:
        if plugin.can_handle_chat_completion(
            messages=prompt.raw(),
            **chat_completion_kwargs,
        ):
            message = plugin.handle_chat_completion(
                messages=prompt.raw(),
                **chat_completion_kwargs,
            )
            if message is not None:
                return message

    # Print full prompt to debug log
    logger.debug(prompt.dump())

    # Route to the appropriate provider
    if is_anthropic_model(model):
        # Anthropic Claude models
        response = ianthropic.create_chat_completion(
            messages=prompt.raw(),
            **chat_completion_kwargs,
        )
    else:
        # OpenAI models
        chat_completion_kwargs.update(config.get_openai_credentials(model))

        if functions:
            chat_completion_kwargs["functions"] = [
                function.schema for function in functions
            ]

        if _is_reasoning_model(model):
            # Known reasoning model: go straight to the compatible kwargs.
            # response_format is kept here — gpt-5 supports it; o1 incompatibility
            # is discovered via the retry below on first failure.
            chat_completion_kwargs["max_completion_tokens"] = chat_completion_kwargs.pop("max_tokens")
            chat_completion_kwargs.pop("temperature", None)

        try:
            response = iopenai.create_chat_completion(
                messages=prompt.raw(),
                **chat_completion_kwargs,
            )
        except openai.error.InvalidRequestError as e:
            err = str(e)
            retried = False
            if "max_tokens" in err and "max_completion_tokens" in err:
                # Newer models (o1, o3, gpt-5-*…) use max_completion_tokens.
                # Cache so subsequent calls skip this path.
                _REASONING_MODELS.add(model)
                logger.debug(
                    f"Model {model} requires max_completion_tokens; caching and retrying."
                )
                chat_completion_kwargs["max_completion_tokens"] = chat_completion_kwargs.pop("max_tokens")
                chat_completion_kwargs.pop("temperature", None)
                retried = True
            if "response_format" in err:
                _NO_RESPONSE_FORMAT_MODELS.add(model)
                logger.debug(
                    f"Model {model} rejects response_format; caching and retrying."
                )
                chat_completion_kwargs.pop("response_format", None)
                retried = True
            if not retried:
                raise
            response = iopenai.create_chat_completion(
                messages=prompt.raw(),
                **chat_completion_kwargs,
            )

    logger.debug(f"Response: {response}")

    if hasattr(response, "error"):
        logger.error(response.error)
        raise RuntimeError(response.error)

    first_message: ResponseMessageDict = response.choices[0].message
    content: str | None = first_message.get("content")
    function_call: FunctionCallDict | None = first_message.get("function_call")

    for plugin in config.plugins:
        if not plugin.can_handle_on_response():
            continue
        # TODO: function call support in plugin.on_response()
        content = plugin.on_response(content)

    if model not in ALL_CHAT_MODELS:
        from autogpt.llm.base import ChatModelInfo
        ALL_CHAT_MODELS[model] = ChatModelInfo(
            name=model,
            prompt_token_cost=0.0,
            completion_token_cost=0.0,
            max_tokens=128000,
            supports_functions=False,
        )

    return ChatModelResponse(
        model_info=ALL_CHAT_MODELS[model],
        content=content,
        function_call=OpenAIFunctionCall(
            name=function_call["name"], arguments=function_call["arguments"]
        )
        if function_call
        else None,
    )
