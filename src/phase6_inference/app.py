"""
Gradio web interface for MiniLLM chat.

Provides an interactive chat UI with adjustable generation parameters
(temperature, top-k, top-p, max tokens).
"""
import argparse
import gradio as gr
import torch
from safetensors.torch import load_file
from typing import List, Tuple, Optional

from src.common.config import MiniLLMConfig
from src.phase3_model.model import MiniLLM
from src.common.utils import load_tokenizer
from src.phase6_inference.generate import TextGenerator

# Global reference to the generator (set at startup)
_generator: Optional[TextGenerator] = None


def load_model(
    model_config_path: str,
    model_path: str,
    tokenizer_path: str,
) -> TextGenerator:
    """
    Load model weights and tokenizer, then create a TextGenerator.

    Args:
        model_config_path: Path to YAML model config.
        model_path: Path to safetensors weights.
        tokenizer_path: Path to tokenizer.json.

    Returns:
        Configured TextGenerator instance.
    """
    config = MiniLLMConfig.from_yaml(model_config_path)
    model = MiniLLM(config)
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    return TextGenerator(model, tokenizer)


def chat_fn(
    message: str,
    history: List[Tuple[str, str]],
    temperature: float,
    top_k: float,
    top_p: float,
    max_tokens: float,
) -> str:
    """
    Gradio chat callback function.

    Builds conversation context from history, generates a response,
    and returns the generated text for the chatbot display.

    Args:
        message: The latest user message.
        history: List of (user_msg, assistant_msg) tuples.
        temperature: Sampling temperature.
        top_k: Top-K filter.
        top_p: Top-P (nucleus) filter.
        max_tokens: Maximum tokens to generate.

    Returns:
        The assistant's generated response as a string.
    """
    global _generator
    if _generator is None:
        return "Error: Model not loaded. Please restart the application."

    # Build conversation context from the last 3 turns
    context = ""
    for turn in history[-3:]:
        context += f"User: {turn[0]}\nAssistant: {turn[1]}\n"
    context += f"User: {message}\nAssistant:"

    result = _generator.generate(
        context,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
    )

    return result["text"].strip()


def main() -> None:
    """
    Entry point: loads the model and launches the Gradio web UI.

    Usage:
        python -m src.phase6_inference.app \
            --model_config configs/model/config_25m.yaml \
            --model_path output/final/model.safetensors \
            --tokenizer_path models/tokenizer/tokenizer.json \
            --port 7860
    """
    parser = argparse.ArgumentParser(description="MiniLLM Chat Demo")
    parser.add_argument(
        "--model_config",
        default="configs/model/config_25m.yaml",
        help="Path to model YAML config",
    )
    parser.add_argument(
        "--model_path",
        default="output/final/model.safetensors",
        help="Path to safetensors model weights",
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

    # Build Gradio interface
    with gr.Blocks(title="MiniLLM Chat", theme=gr.themes.Soft()) as demo:
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
            # Append user message to history
            chatbot_history.append((message, ""))
            response = chat_fn(message, chatbot_history[:-1], temp, t_k, t_p, m_tok)
            # Update the last entry with the assistant response
            chatbot_history[-1] = (message, response)
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
