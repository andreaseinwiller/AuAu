import httpx
import tqdm
import asyncio

from datetime import datetime
from typing import Any, Literal, TypedDict, cast
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from vllm import LLM, SamplingParams, TokensPrompt

from llm_audit.util import (
    ErrorDict,
    ExperimentType,
    get_enable_thinking,
    get_formatted_text_content,
)


class ResponseMeta(TypedDict):
    """
    Metadata of LLM inference.
    """

    model_name: str
    temperature: float
    seed: int | None
    system_prompts: list[str]
    user_prompts: list[str]
    response: str | None
    start_time: str | None
    end_time: str | None
    prompt_tokens: int  # input tokens
    completion_tokens: int  # output tokens (including reasoning)
    total_tokens: int  # input + output tokens
    error: ErrorDict | None


class AsyncGenerationConfig(TypedDict):
    max_concurrent: int


def generate(
    experiment_type: ExperimentType,
    client: OpenAI | None,
    model_name: str,
    model_format: Literal["text-only", "multimodal"],
    model_setup: Literal["OpenRouter", "vLLM", "Alternative"],
    sampling_params: SamplingParams | None,
    model: LLM | None,
    tokenizer: Any | None,
    temperature: float,
    seed: int | None,
    repetitions: int,
    system_prompts: list[str],
    batch_user_prompts: list[list[str]],
    use_tqdm: bool = False,
    use_async: bool = False,
    async_config: AsyncGenerationConfig | None = None,
    extra_body: dict | None = None,  # type: ignore
) -> list[ResponseMeta]:
    """
    Generates an LLM response given system and user prompts using OpenRouter.

    This function constructs a conversation by combining multiple system and user prompts,
    formatted according to the specified model format. It sends the messages to the LLM
    using the provided OpenAI client and returns the respective metadata.

    Args:
        experiment_type (ExperimentType): Type to differentiate between different experiment runs such as "closed_question" and "open_question".
        client (OpenAI | None): An instance of the OpenAI client used to send the request.
        model_name (str): The model label defined by OpenRouter.
        model_format (str): The format laybel used to determine the prompt structure (e.g., "text-only", or "multimodal").
        model_setup (str): key to determine inference setup (e.g., "OpenRouter" or "vLLM").
        sampling_params (SamplingParams | None): <TODO  description>
        model (LLM | None): <TODO description>
        tokenizer (Any | None): <TODO description>
        temperature (float): Set focus on determinism or randomness.
        seed (int | None): Attempt deterministic sampling, which is not not guaranteed according to OpenAI. Can be set to None to enable repetition output variance during vLLM continuous batch inference.
        repetitions (int): Number of reruns.
        system_prompts (list[str]): A list of system prompts to guide the model.
        batch_user_prompts (list[list[str]]): A batch of user prompts related to the task. Each list inside the outer list will be one conversation.
        use_tqdm (bool): <TODO description>
        use_async (bool): <TODO description>
        async_config (AsyncGenerationConfig | None): <TODO description>
        extra_body (dict | None): <TODO description>

    Returns:
        list[ResponseMeta]: A list of dictionaries containing metadata about the response generation step,
        including the model name, system and user prompts, the generated response,
        start and end timestamps of the generation, and any error message.
    """
    response_metas: list[ResponseMeta] = [
        {
            "model_name": model_name,
            "temperature": temperature,
            "seed": (seed + i) if seed is not None else None,
            "system_prompts": system_prompts,
            "user_prompts": user_prompts,
            "response": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "start_time": None,
            "end_time": None,
            "error": None,
        }
        for i in range(repetitions)
        for user_prompts in batch_user_prompts
    ]

    # Create list of messages (a conversation) for the full batch of inputs
    convos: list[list[ChatCompletionMessageParam]] = []
    for response_meta in response_metas:
        try:
            messages = []

            for system_prompt in response_meta["system_prompts"]:
                messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {
                            "role": "system",
                            "content": get_formatted_text_content(model_format=model_format, content=system_prompt),
                        },
                    )
                )
            for user_prompt in response_meta["user_prompts"]:
                messages.append(
                    cast(
                        ChatCompletionMessageParam,
                        {
                            "role": "user",
                            "content": get_formatted_text_content(model_format=model_format, content=user_prompt),
                        },
                    )
                )
                convos.append(messages)
        except Exception as e:
            # E.g. ValueError for invalid model_format
            response_meta["error"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            convos.append([])

    if model_setup in ["OpenRouter", "Alternative"] and not use_async:
        iterator = (
            tqdm.tqdm(zip(response_metas, convos), total=len(batch_user_prompts) * repetitions)
            if use_tqdm
            else zip(response_metas, convos)
        )
        for response_meta, convo in iterator:
            # This would be a lot faster with async, but also a hassle to implement
            response_text = None
            try:
                if client is None:
                    raise ValueError(
                        f"OpenAI API setup not initialized, which is required to perform inference with the {model_setup} API."
                    )
                start_time = datetime.now().isoformat()
                response = client.chat.completions.create(
                    model=model_name,
                    messages=convo,
                    extra_body=extra_body
                    or {
                        "chat_template_kwargs": {
                            "enable_thinking": get_enable_thinking(experiment_type=experiment_type)
                        }
                    },
                    temperature=temperature,
                    seed=response_meta["seed"],
                    timeout=60,
                )
                if (
                    response is None
                    or not response.choices
                    or not response.choices[0].message
                    or not response.choices[0].message.content
                    or not response.usage
                    or not response.usage.prompt_tokens
                    or not response.usage.completion_tokens
                    or not response.usage.total_tokens
                ):
                    raise ValueError(f"Invalid {model_setup} API response.")
                response_text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                end_time = datetime.now().isoformat()
                response_meta.update(
                    {
                        "response": response_text,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )
            except Exception as e:
                # E.g. ValueError for invalid model_format
                response_meta["error"] = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),  # This key might be "message" in old computation results!
                }
    elif model_setup in ["OpenRouter", "Alternative"] and use_async:
        assert async_config, "No async configuration passed (e.g., max concurrent requests)"
        assert client, "OpenAI client not initialized"
        response_metas = asyncio.run(
            process_batch_async(
                sync_client=client,
                model_name=model_name,
                experiment_type=experiment_type,
                convos=convos,
                response_metas=response_metas,
                temperature=temperature,
                config=async_config,
                use_tqdm=use_tqdm,
                extra_body=extra_body,
            )
        )
    elif model_setup == "vLLM":
        assert model is not None
        assert tokenizer is not None

        tokenized_prompts = []
        for messages in convos:
            start_time = datetime.now().isoformat()
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=get_enable_thinking(experiment_type=experiment_type),
                )
            except ValueError:
                # Base model throws ValueError; trigger alternative approach to encode (input) text into tokens
                input_ids = tokenizer.encode("\n\n".join(cast(str, m["content"]) for m in messages))
            tokenized_prompts.append(TokensPrompt({"prompt_token_ids": input_ids}))

        outputs = model.generate(
            prompts=tokenized_prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
        for output, response_meta, tokenized_prompt in zip(outputs, response_metas, tokenized_prompts):
            response_text = output.outputs[0].text

            reasoning_tokens = 0
            if hasattr(output.outputs[0], "reasoning_token_ids"):
                reasoning_tokens = len(output.outputs[0].reasoning_token_ids)
            elif hasattr(output, "usage") and hasattr(output.usage, "reasoning_tokens"):
                reasoning_tokens = output.usage.reasoning_tokens
            elif hasattr(output.outputs[0], "extended_thinking_tokens"):
                reasoning_tokens = output.outputs[0].extended_thinking_tokens

            prompt_tokens = len(tokenized_prompt["prompt_token_ids"])
            completion_tokens = len(output.outputs[0].token_ids) + reasoning_tokens
            total_tokens = prompt_tokens + completion_tokens
            end_time = datetime.now().isoformat()
            response_meta.update(
                {
                    "response": response_text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
    return response_metas


async def process_batch_async(
    sync_client: OpenAI,
    model_name: str,
    experiment_type: ExperimentType,
    convos: list[list[ChatCompletionMessageParam]],
    response_metas: list[ResponseMeta],
    temperature: float,
    config: AsyncGenerationConfig,
    use_tqdm: bool,
    extra_body: dict | None = None,  # type: ignore
) -> list[ResponseMeta]:
    semaphore = asyncio.Semaphore(config["max_concurrent"])
    # By default only 100 tcp connections can be opened. This means config["max_concurrent"] becomes ineffective for larger values.
    # Since only 100 connections may be opened, but the semaphore is larger, the requests will timeout, should the other requests not be done in time.
    # We are using our own http client with higher limits to work around this
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=config["max_concurrent"],
            max_keepalive_connections=config["max_concurrent"],
        )
    )
    async_client = AsyncOpenAI(
        base_url=sync_client.base_url,
        api_key=sync_client.api_key,
        http_client=http_client,
    )

    async def process_one(response_meta: ResponseMeta, convo: list[ChatCompletionMessageParam]) -> ResponseMeta:
        async with semaphore:
            try:
                start_time = datetime.now().isoformat()
                response = await async_client.chat.completions.create(
                    model=model_name,
                    messages=convo,
                    extra_body=extra_body
                    or {
                        "chat_template_kwargs": {
                            "enable_thinking": get_enable_thinking(experiment_type=experiment_type)
                        }
                    },
                    temperature=temperature,
                    seed=response_meta["seed"],
                    timeout=60,
                )
                if (
                    response is None
                    or not response.choices
                    or not response.choices[0].message
                    or not response.choices[0].message.content
                    or not response.usage
                    or not response.usage.prompt_tokens
                    or not response.usage.completion_tokens
                    or not response.usage.total_tokens
                ):
                    raise ValueError("Invalid API response.")
                response_text = response.choices[0].message.content
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                end_time = datetime.now().isoformat()
                response_meta.update(
                    {
                        "response": response_text,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                )
            except Exception as e:
                # E.g. ValueError for invalid model_format
                response_meta["error"] = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),  # This key might be "message" in old computation results!
                }
        return response_meta

    tasks = [process_one(response_meta, convo) for response_meta, convo in zip(response_metas, convos)]

    if use_tqdm:
        response_metas = await tqdm.asyncio.tqdm.gather(*tasks)
    else:
        response_metas = await asyncio.gather(*tasks)
    return response_metas
