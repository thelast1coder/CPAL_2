"""Prompt construction shared by every method (identical prompts across methods, PROJECT.md 12)."""
from __future__ import annotations

GSM8K_SUFFIX = "\nPlease reason step by step, and put your final answer after '####'."


def gsm8k_user_message(question: str) -> str:
    return question.strip() + GSM8K_SUFFIX


def chat_prompt(tokenizer, user_message: str, model_family: str) -> str:
    """Apply the model's chat template with a generation prompt.

    For Qwen3 the template takes `enable_thinking=False` for non-thinking mode (verify against the
    downloaded tokenizer_config.json chat template on the VM).
    """
    msgs = [{"role": "user", "content": user_message}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if model_family == "qwen3_moe":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(msgs, **kwargs)


def math_chat_example(tokenizer, question: str, answer: str) -> str:
    """MetaMathQA calibration example in the chat template: question as user, answer as assistant."""
    msgs = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return tokenizer.apply_chat_template(msgs, tokenize=False)
