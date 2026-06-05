"""
Prompt templates for SFT data formatting.
Supports: alpaca, chatml, llama
"""

from typing import Any

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------

ALPACA_TEMPLATE: dict[str, str] = {
    "system": "",  # Alpaca has no system prompt
    "user_start": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
    ),
    "user_end": "\n\n### Response:\n",
    "assistant_end": "",  # EOS token is appended separately
}

ALPACA_TEMPLATE_WITH_INPUT: dict[str, str] = {
    "system": "",
    "user_start": (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n"
    ),
    "input_start": "\n\n### Input:\n",
    "user_end": "\n\n### Response:\n",
    "assistant_end": "",
}

CHATML_TEMPLATE: dict[str, str] = {
    "system_start": "<|im_start|>system\n",
    "system_end": "<|im_end|>\n",
    "user_start": "<|im_start|>user\n",
    "user_end": "<|im_end|>\n",
    "assistant_start": "<|im_start|>assistant\n",
    "assistant_end": "<|im_end|>\n",
}

LLAMA_TEMPLATE: dict[str, str] = {
    "system_start": "<s>[INST] <<SYS>>\n",
    "system_end": "\n<</SYS>>\n\n",
    "user_start": "",  # combined with system or standalone
    "user_end": " [/INST] ",
    "assistant_end": " </s>",
}

# For llama, the format is:
# With system: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST] {response} </s>
# Without system: <s>[INST] {instruction} [/INST] {response} </s>

# ---------------------------------------------------------------------------
# PromptTemplate class
# ---------------------------------------------------------------------------

class PromptTemplate:
    """Unified interface for different prompt templates."""

    def __init__(self, template_type: str = "alpaca") -> None:
        self.template_type: str = template_type
        self._template: dict[str, str] = self._load_template(template_type)

    def _load_template(self, template_type: str) -> dict[str, str]:
        """Return the raw template dictionary for the given type."""
        if template_type == "alpaca":
            return dict(ALPACA_TEMPLATE)
        elif template_type == "chatml":
            return dict(CHATML_TEMPLATE)
        elif template_type == "llama":
            return dict(LLAMA_TEMPLATE)
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    @property
    def template(self) -> dict[str, str]:
        return self._template

    def format_instruction(
        self,
        instruction: str,
        input_text: str = "",
        system_prompt: str = "",
    ) -> tuple[str, str]:
        """
        Format an instruction into a prompt and a response prefix.

        Returns:
            (full_prompt, response_prefix)

            - full_prompt: everything BEFORE the assistant response.
            - response_prefix: the assistant marker that starts the response
              (usually empty because it is already part of full_prompt).

        During training labels, everything in full_prompt is masked (set to -100).
        """
        if self.template_type == "alpaca":
            prompt = self._format_alpaca(instruction, input_text)
            response_prefix = ""

        elif self.template_type == "chatml":
            prompt = self._format_chatml(instruction, input_text, system_prompt)
            response_prefix = ""

        elif self.template_type == "llama":
            prompt = self._format_llama(instruction, input_text, system_prompt)
            response_prefix = ""

        else:
            raise ValueError(f"Unknown template: {self.template_type}")

        return prompt, response_prefix

    def format_full(
        self,
        instruction: str,
        input_text: str = "",
        output_text: str = "",
        system_prompt: str = "",
    ) -> str:
        """Format a complete instruction-response pair, including EOS markers."""
        prompt, _resp_prefix = self.format_instruction(
            instruction, input_text, system_prompt
        )

        if self.template_type == "alpaca":
            return prompt + output_text  # EOS added by tokenizer post_processor
        elif self.template_type == "chatml":
            return prompt + output_text + "<|im_end|>\n"
        elif self.template_type == "llama":
            return prompt + output_text + " </s>"
        else:
            return prompt + output_text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_alpaca(self, instruction: str, input_text: str) -> str:
        """Build the Alpaca-style prompt (everything before the response)."""
        if input_text and input_text.strip():
            return (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            return (
                "Below is an instruction that describes a task. Write a response "
                "that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

    def _format_chatml(
        self, instruction: str, input_text: str, system_prompt: str
    ) -> str:
        """Build the ChatML-style prompt."""
        parts: list[str] = []

        # System message
        if system_prompt and system_prompt.strip():
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        else:
            parts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")

        # User message
        parts.append(f"<|im_start|>user\n{instruction}")
        if input_text and input_text.strip():
            parts.append(f"\n{input_text}")
        parts.append("<|im_end|>\n")

        # Assistant start (response prefix)
        parts.append("<|im_start|>assistant\n")

        return "".join(parts)

    def _format_llama(
        self, instruction: str, input_text: str, system_prompt: str
    ) -> str:
        """Build the LLaMA-style prompt."""
        if system_prompt and system_prompt.strip():
            prompt = (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction}"
            )
        else:
            prompt = f"<s>[INST] {instruction}"

        if input_text and input_text.strip():
            prompt += f"\n{input_text}"

        prompt += " [/INST] "
        return prompt

# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_template(template_type: str) -> PromptTemplate:
    """Create a PromptTemplate by name."""
    return PromptTemplate(template_type)
