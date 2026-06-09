"""
Gradio web interface for MiniLLM chat.

Provides an interactive chat UI with adjustable generation parameters
(temperature, top-k, top-p, max tokens).

Usage:
    python -m src.phase6_inference.app \
        --model_config configs/model/config_25m.yaml \
        --model_path models/checkpoints/pretrain_best.pt
"""
import argparse
import gradio as gr
import torch
from typing import List, Dict, Optional

from src.common.config import MiniLLMConfig
from src.phase3_model.model import MiniLLM
from src.common.utils import load_tokenizer
from src.phase6_inference.generate import TextGenerator
from src.phase5_sft.templates import PromptTemplate

# Global reference to the generator (set at startup)
_generator: Optional[TextGenerator] = None


def load_model(
    model_config_path: str,
    model_path: str,
    tokenizer_path: str,
) -> TextGenerator:
    """Load model weights and tokenizer, then create a TextGenerator."""
    config = MiniLLMConfig.from_yaml(model_config_path)
    model = MiniLLM(config)
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(model_path)
    else:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    return TextGenerator(model, tokenizer)


def chat_fn(
    message: str,
    history: List[Dict[str, str]],
    temperature: float,
    top_k: int,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    Build an Alpaca-format prompt and generate a response.

    Args:
        message: The latest user message.
        history: Previous messages in Gradio 6.x format.
        temperature: Sampling temperature.
        top_k: Top-K filter.
        top_p: Top-P (nucleus) filter.
        max_tokens: Maximum tokens to generate.

    Returns:
        Generated response string.
    """
    global _generator
    if _generator is None:
        return "Error: Model not loaded."

    # Build multi-turn context as Alpaca "input" field
    input_text = ""
    recent = history[-6:]  # up to 3 turns
    for msg in recent:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        input_text += f"{role_label}: {msg['content']}\n"

    template = PromptTemplate("alpaca")
    prompt, _ = template.format_instruction(
        instruction=message,
        input_text=input_text.strip(),
    )

    result = _generator.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    response = result["text"].strip()
    if not response:
        response = "(model generated empty response)"
    return response


def main() -> None:
    """Entry point: loads the model and launches the Gradio web UI."""
    parser = argparse.ArgumentParser(description="MiniLLM Chat Demo")
    parser.add_argument(
        "--model_config",
        default="configs/model/config_25m.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--model_path",
        default="models/checkpoints/pretrain_best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        default="models/tokenizer/tokenizer.json",
        help="Path to tokenizer JSON file",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Share the demo publicly via Gradio temporary link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the web server on",
    )
    args = parser.parse_args()

    global _generator
    print("Loading model...")
    _generator = load_model(args.model_config, args.model_path, args.tokenizer_path)
    print("Model loaded successfully!")

    # Build Gradio interface with parameter sliders
    with gr.Blocks(title="MiniLLM Chat") as demo:
        gr.Markdown("# MiniLLM Chat")
        gr.Markdown("A small language model trained from scratch on CPU.")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Conversation")
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    label="Message",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                temperature = gr.Slider(
                    0.0, 2.0, value=0.8, step=0.1, label="Temperature"
                )
                top_k = gr.Slider(1, 100, value=50, step=1, label="Top-K")
                top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-P")
                max_tokens = gr.Slider(
                    16, 512, value=128, step=16, label="Max Tokens"
                )

        # ---- Event handlers ----
        def _respond(message, chatbot_history, temp, t_k, t_p, m_tok):
            """Handle one turn: append user msg, generate, append assistant msg."""
            chatbot_history.append({"role": "user", "content": message})
            response = chat_fn(
                message, chatbot_history[:-1],
                float(temp), int(t_k), float(t_p), int(m_tok),
            )
            chatbot_history.append({"role": "assistant", "content": response})
            return chatbot_history

        submit_btn.click(
            _respond,
            [msg, chatbot, temperature, top_k, top_p, max_tokens],
            [chatbot],
            queue=True,
        ).then(lambda: "", None, [msg])

        msg.submit(
            _respond,
            [msg, chatbot, temperature, top_k, top_p, max_tokens],
            [chatbot],
            queue=True,
        ).then(lambda: "", None, [msg])

        clear_btn.click(lambda: [], None, [chatbot])

    print(f"Starting Gradio server on port {args.port}...")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
