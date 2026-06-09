"""
Text generation with multiple decoding strategies.

Supports temperature sampling, top-k filtering, top-p (nucleus) filtering,
repetition penalty, and greedy decoding.
"""
import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any


class TextGenerator:
    """
    Text generator with support for various decoding strategies.

    Attributes:
        model: The language model (nn.Module with a `generate` method or forward).
        tokenizer: The tokenizer instance (HuggingFace Tokenizers or compatible).
        eos_token_id: Token ID that signals end-of-sequence.
        bos_token_id: Token ID that signals beginning-of-sequence.
        pad_token_id: Token ID used for padding.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any) -> None:
        """
        Initialize the text generator.

        Args:
            model: Trained MiniLLM model.
            tokenizer: Trained tokenizer (HuggingFace Tokenizers or compatible).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = (
            tokenizer.token_to_id("<EOS>") or getattr(model.config, "eos_token_id", None)
        )
        self.bos_token_id = (
            tokenizer.token_to_id("<BOS>") or getattr(model.config, "bos_token_id", None)
        )
        self.pad_token_id = (
            tokenizer.token_to_id("<PAD>") or getattr(model.config, "pad_token_id", None)
        )
        self.unk_token_id = (
            tokenizer.token_to_id("<UNK>") or getattr(model.config, "unk_token_id", None)
        )

        # Token IDs that should never be generated (always suppressed)
        self._forbidden_ids: set[int] = set()
        for tid in (self.pad_token_id, self.bos_token_id, self.unk_token_id):
            if tid is not None:
                self._forbidden_ids.add(tid)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_tokens: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt using the configured decoding strategy.

        Args:
            prompt: The input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_k: Keep only the k most likely tokens (0 = disabled).
            top_p: Nucleus sampling threshold (1.0 = disabled).
            repetition_penalty: Penalize tokens that already appear (1.0 = disabled).
            do_sample: If False, use argmax (greedy) instead of multinomial sampling.
            stop_tokens: Optional list of strings; generation stops if any appear
                         in the recently generated text.

        Returns:
            A dict with:
                - text: str -- the generated text only (without the prompt).
                - full_text: str -- prompt + generated text.
                - tokens: List[int] -- the generated token IDs.
                - num_tokens: int -- number of tokens generated.
                - truncated: bool -- whether generation hit max_new_tokens before EOS.
        """
        # ========================
        # 1. Encode the prompt
        # ========================
        encoded = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=next(self.model.parameters()).device)

        generated_ids: List[int] = []
        truncated: bool = False

        max_seq_len = getattr(self.model.config, "max_seq_len", 512)

        # ========================
        # 2. Auto-regressive loop
        # ========================
        for _step in range(max_new_tokens):
            # ---- Truncate context if it exceeds max_seq_len ----
            if input_ids.size(1) > max_seq_len:
                input_ids = input_ids[:, -max_seq_len:]

            # ---- Forward pass ----
            logits = self.model(input_ids)  # shape: (1, seq_len, vocab_size)
            next_logits = logits[:, -1, :]   # shape: (1, vocab_size)

            # ---- Temperature ----
            if temperature <= 0.0:
                # Greedy decoding -- skip all filtering
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                next_token_id = next_token.item()
                generated_ids.append(next_token_id)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token_id == self.eos_token_id:
                    break
                continue

            next_logits = next_logits / temperature

            # ---- Repetition penalty ----
            if repetition_penalty != 1.0:
                seen_token_ids = set(input_ids[0].tolist())
                for token_id in seen_token_ids:
                    if next_logits[0, token_id] < 0:
                        next_logits[0, token_id] *= repetition_penalty
                    else:
                        next_logits[0, token_id] /= repetition_penalty

            # ---- Top-K filtering ----
            if top_k > 0:
                top_k_actual = min(top_k, next_logits.size(-1))
                topk_vals, _ = torch.topk(next_logits, top_k_actual)
                threshold = topk_vals[:, -1:]  # smallest value among top-k
                next_logits[next_logits < threshold] = float("-inf")

            # ---- Top-P (nucleus) filtering ----
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Mask tokens whose cumulative probability exceeds top_p
                sorted_mask = cum_probs > top_p
                # Shift mask to keep at least one token
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False

                # Scatter mask back to original order
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits[indices_to_remove] = float("-inf")

            # ---- Suppress forbidden tokens (PAD, BOS, UNK should never be generated) ----
            for tid in self._forbidden_ids:
                next_logits[0, tid] = float("-inf")

            # ---- Sampling ----
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()

            # ---- Check for EOS ----
            if next_token_id == self.eos_token_id:
                generated_ids.append(next_token_id)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                break

            # ---- Check stop tokens ----
            if stop_tokens:
                generated_ids.append(next_token_id)
                # Decode the most recent tokens to check for stop strings
                recent_text = self.tokenizer.decode(generated_ids[-10:])
                should_stop = False
                for stop in stop_tokens:
                    if stop in recent_text:
                        should_stop = True
                        break
                if should_stop:
                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    break
                # Undo the append that was just for checking
                generated_ids.pop()

            generated_ids.append(next_token_id)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        else:
            # Loop completed without break -> truncated
            truncated = True

        # ========================
        # 3. Decode the results
        # ========================
        full_text = self.tokenizer.decode(input_ids[0].tolist())
        prompt_text = self.tokenizer.decode(encoded.ids)

        # Extract only the generated part
        if full_text.startswith(prompt_text):
            generated_text = full_text[len(prompt_text):]
        else:
            # Fallback: the decode might not be a simple prefix match
            generated_text = self.tokenizer.decode(generated_ids)

        return {
            "text": generated_text,
            "full_text": full_text,
            "tokens": generated_ids,
            "num_tokens": len(generated_ids),
            "truncated": truncated,
        }

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple prompts sequentially.

        Args:
            prompts: List of prompt strings.
            **kwargs: Passed through to `self.generate`.

        Returns:
            List of result dicts, one per prompt.
        """
        return [self.generate(p, **kwargs) for p in prompts]


def chat_loop(
    model: torch.nn.Module,
    tokenizer: Any,
    max_tokens: int = 256,
) -> None:
    """
    Interactive chat loop for terminal testing.

    Commands:
        Type 'quit' or 'exit' to stop.
        Type 'clear' to reset the conversation history.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        max_tokens: Maximum tokens per response.
    """
    generator = TextGenerator(model, tokenizer)

    print("=" * 60)
    print("MiniLLM Chat - Type 'quit' to exit, 'clear' to reset")
    print("=" * 60)

    history: List[Dict[str, str]] = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history = []
            print("Chat history cleared.")
            continue
        if not user_input:
            continue

        # Build context from the last 3 conversation turns
        context = ""
        for turn in history[-3:]:
            context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        context += f"User: {user_input}\nAssistant:"

        result = generator.generate(context, max_new_tokens=max_tokens)
        response = result["text"].strip()

        print(f"\nAssistant: {response}")

        history.append({"user": user_input, "assistant": response})
