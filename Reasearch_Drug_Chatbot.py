from __future__ import annotations

import copy
import hmac
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

try:
    from peft import PeftModel
except ImportError as exc:
    raise ImportError(
        "peft is required for adapter serving. Install it into the serving env "
        "with: pip install peft"
    ) from exc

__all__ = [
    "ServingConfig",
    "GemmaRagEngine",
    "build_generation_config",
    "format_rag_messages",
    "load_user_credentials",
    "make_auth_callback",
    "configure_logging",
]

LOGGER = logging.getLogger("gemma_serving")

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a drug-information and clinical-research assistant. Answer questions "
    "about medications, clinical trials, mechanisms, interactions, effects, and "
    "research findings clearly and accurately. Be specific: name the drugs, classes, "
    "mechanisms, and study findings relevant to the question. Where the literature is "
    "uncertain or mixed, say so. Close with a brief note that this is educational "
    "information, not individualized medical advice, and that clinical decisions "
    "belong to a licensed clinician or pharmacist."
)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stream handler if logging has not been configured yet."""
    if logging.getLogger().handlers:
        LOGGER.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@dataclass(frozen=True)
class ServingConfig:
    """Inference configuration. All fields are overridable via environment."""

    base_model_id: str = os.environ.get("RAG_BASE_MODEL", "google/gemma-4-12B-it")
    adapter_dir: str = os.environ.get("RAG_ADAPTER_DIR", "gemma-4-12b-drugbot-lora/")
    device: str = os.environ.get("RAG_DEVICE", "cuda")
    # sdpa is the supported fast path for Gemma 4. flash_attention_2 crashes on
    # the global-attention layers (head_dim 512). eager is the safe fallback.
    attn_implementation: str = os.environ.get("RAG_ATTN", "sdpa")
    merge_adapter: bool = os.environ.get("RAG_MERGE_ADAPTER", "0") == "1"
    max_new_tokens: int = int(os.environ.get("RAG_MAX_NEW_TOKENS", "512"))
    # Official Gemma sampling values. Inference only, not training params.
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64


def build_generation_config(
    base_generation_config: GenerationConfig, config: ServingConfig
) -> GenerationConfig:
    """Clone the checkpoint generation config and apply the official sampling values.

    Starting from the checkpoint config preserves the correct eos and pad token ids
    rather than guessing Gemma special tokens by hand.
    """
    gen = GenerationConfig.from_dict(base_generation_config.to_dict())
    gen.do_sample = True
    gen.temperature = config.temperature
    gen.top_p = config.top_p
    gen.top_k = config.top_k
    gen.max_new_tokens = config.max_new_tokens
    if gen.pad_token_id is None and gen.eos_token_id is not None:
        eos = gen.eos_token_id
        gen.pad_token_id = eos if isinstance(eos, int) else eos[0]
    return gen


def format_rag_messages(
    question: str,
    context_blocks: Sequence[str],
    system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
) -> list[dict]:
    """Fold guidance and retrieved context into a single user turn.

    Gemma chat templates are most compatible when instructions and context live in
    the user turn rather than a separate system role.
    """
    if context_blocks:
        context = "\n\n".join(
            f"[{i + 1}] {block}" for i, block in enumerate(context_blocks)
        )
        user_content = (
            f"{system_instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    else:
        user_content = f"{system_instruction}\n\nQuestion: {question}"
    return [{"role": "user", "content": user_content}]


class GemmaRagEngine:
    """Wraps a base Gemma 4 model and a LoRA adapter for text-only RAG inference."""

    def __init__(self, config: ServingConfig) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self.generation_config: GenerationConfig | None = None
        self.device = torch.device(config.device)

    def load(self) -> "GemmaRagEngine":
        self._validate_environment()
        self._load_processor()
        self._load_model()
        self.generation_config = build_generation_config(
            self.model.generation_config, self.config
        )
        LOGGER.info(
            "Engine ready. base=%s adapter=%s device=%s dtype=%s attn=%s merged=%s",
            self.config.base_model_id,
            self.config.adapter_dir,
            self.device,
            self.model.dtype,
            self.config.attn_implementation,
            self.config.merge_adapter,
        )
        return self

    def _validate_environment(self) -> None:
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. Check the "
                "WSL2 GPU passthrough and that the torch build matches your driver."
            )
        adapter = Path(self.config.adapter_dir)
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"No adapter_config.json under {adapter.resolve()}. "
                "PeftModel.from_pretrained needs a PEFT-format adapter directory "
                "(adapter_config.json plus adapter_model.safetensors). If Train_loRa.py "
                "produced a KerasHub .npz, re-export the adapter in PEFT format first."
            )

    def _select_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _device_map(self) -> dict | str:
        if self.config.device.startswith("cuda"):
            index = self.device.index if self.device.index is not None else 0
            return {"": index}
        return {"": "cpu"}

    def _load_processor(self) -> None:
        LOGGER.info("Loading processor for %s", self.config.base_model_id)
        self.processor = AutoProcessor.from_pretrained(
            self.config.base_model_id, padding_side="left"
        )

    def _load_model(self) -> None:
        dtype = self._select_dtype()
        LOGGER.info(
            "Loading base model %s (dtype=%s, attn=%s)",
            self.config.base_model_id,
            dtype,
            self.config.attn_implementation,
        )
        try:
            base = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_id,
                dtype=dtype,  # older transformers use torch_dtype
                attn_implementation=self.config.attn_implementation,
                device_map=self._device_map(),
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "Out of memory loading the base model. Try the E4B variant instead of "
                "12B, or enable 4-bit loading."
            ) from exc

        LOGGER.info("Attaching LoRA adapter from %s", self.config.adapter_dir)
        try:
            model = PeftModel.from_pretrained(base, self.config.adapter_dir)
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Failed to attach the LoRA adapter. A common cause is a module name "
                "mismatch between the class used in Train_loRa.py and AutoModelForCausalLM. "
                "Load the base with the same class you trained against, or re-export the "
                f"adapter with matching target_modules. Original error: {exc}"
            ) from exc

        if self.config.merge_adapter:
            LOGGER.info("Merging adapter into base weights for faster inference")
            model = model.merge_and_unload()

        model.eval()
        self.model = model

    @torch.inference_mode()
    def generate(
        self, messages: Sequence[dict], max_new_tokens: int | None = None
    ) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Engine not loaded. Call load() before generate().")

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[-1]

        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config = copy.deepcopy(gen_config)
            gen_config.max_new_tokens = max_new_tokens

        start = time.perf_counter()
        try:
            output = self.model.generate(**inputs, generation_config=gen_config)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "Out of memory during generation. Lower max_new_tokens or trim the "
                "retrieved context size."
            ) from exc

        elapsed = time.perf_counter() - start
        new_tokens = int(output.shape[-1] - input_len)
        rate = new_tokens / elapsed if elapsed > 0 else 0.0
        LOGGER.info("Generated %d tokens in %.2fs (%.1f tok/s)", new_tokens, elapsed, rate)

        decode = getattr(self.processor, "decode", None) or self.processor.tokenizer.decode
        answer = decode(output[0][input_len:], skip_special_tokens=True)
        return answer.strip()


def load_user_credentials(env_var: str = "RAG_CHATBOT_USERS") -> dict[str, str]:
    """Parse credentials from an env var formatted as user:pass,user:pass.

    Keeps secrets out of the project tree. Returns a loud default if unset so the
    app still starts in development, but warns to set real credentials.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        LOGGER.warning(
            "No %s set. Falling back to a single default account. Set %s before "
            "exposing the app over Tailscale.",
            env_var,
            env_var,
        )
        return {"capstone": "change-me"}

    creds: dict[str, str] = {}
    for pair in raw.split(","):
        if ":" not in pair:
            continue
        user, password = pair.split(":", 1)
        user = user.strip()
        if user:
            creds[user] = password
    if not creds:
        raise ValueError(f"{env_var} was set but no valid user:pass pairs were parsed.")
    return creds


def make_auth_callback(credentials: dict[str, str]) -> Callable[[str, str], bool]:
    """Return a constant-time Gradio auth callback over the credential map."""

    def verify(username: str, password: str) -> bool:
        expected = credentials.get(username)
        if expected is None:
            # Burn a comparison to keep timing uniform for unknown users.
            hmac.compare_digest(password, password)
            return False
        return hmac.compare_digest(password, expected)

    return verify


def build_demo(engine: GemmaRagEngine):
    """Reference Gradio UI. Adapt this call into the existing app, do not replace it."""
    import gradio as gr

    def respond(message: str, history: list) -> str:
        messages = format_rag_messages(question=message, context_blocks=[])
        return engine.generate(messages)

    return gr.ChatInterface(fn=respond, title="Drug Information RAG Chatbot")


def main() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    configure_logging()

    engine = GemmaRagEngine(ServingConfig()).load()
    demo = build_demo(engine)
    credentials = load_user_credentials()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("RAG_PORT", "7860")),
        auth=make_auth_callback(credentials),
        auth_message="CSC525 drug-information RAG chatbot. Authorized users only.",
        share=False,
        show_error=False,  # set True only while debugging
    )


if __name__ == "__main__":
    main()