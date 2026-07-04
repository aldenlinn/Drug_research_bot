from __future__ import annotations

import faulthandler
import hashlib
import logging
import os
import sys

import torch

# On a fatal signal (e.g. a native-extension segfault mid-training) print a Python traceback to
# stderr, so the crash names the exact call site instead of a bare "Segmentation fault".
faulthandler.enable()
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig

from drugbot_prompts import SYSTEM_PROMPT, base_model_id

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOG = logging.getLogger("train_lora")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --- configuration (all overridable via environment) ---
MODEL_ID = base_model_id()                                   # RAG_BASE_MODEL, shared with serving
INSTRUCTIONS = os.environ.get("DRUGBOT_PAIRS", "data/combined_pairs.jsonl")
OUTPUT_DIR = os.environ.get("DRUGBOT_OUTPUT_DIR", "gemma-4-12b-drugbot-lora")

# 8-bit (LLM.int8) is mandatory: this trainer loads the base in int8 or it does not run.
# There is no bf16 / 4-bit path. The preflight requires a working bitsandbytes CUDA kernel.
SEQ_LEN = env_int("DRUGBOT_SEQ_LEN", 1024)
TRAIN_ROWS = env_int("DRUGBOT_TRAIN_ROWS", 0)                # 0 = use all rows
EVAL_ROWS = env_int("DRUGBOT_EVAL_ROWS", 500)               # cap eval so it does not dominate
EVAL_ENABLED = env_flag("DRUGBOT_EVAL", True)               # 0 = skip eval entirely (fastest; trains on all rows)
EVAL_STEPS = env_int("DRUGBOT_EVAL_STEPS", 500)             # eval cadence (was 250); higher = less overhead
SAVE_STEPS = env_int("DRUGBOT_SAVE_STEPS", 500)             # checkpoint cadence
MAX_STEPS = env_int("DRUGBOT_MAX_STEPS", 0)                  # >0 overrides epochs (token budget)
EPOCHS = env_float("DRUGBOT_EPOCHS", 1.0)
BATCH_SIZE = env_int("DRUGBOT_BATCH_SIZE", 4)
GRAD_ACCUM = env_int("DRUGBOT_GRAD_ACCUM", 4)
GRAD_CKPT = env_flag("DRUGBOT_GRAD_CKPT", True)             # off ~= 20-30% faster if VRAM allows (watch for OOM)
DATALOADER_WORKERS = env_int("DRUGBOT_DATALOADER_WORKERS", 0)  # 0 = no forked workers (avoids
# forked-worker-plus-CUDA segfaults; data is pre-tokenized so collation is trivial anyway)
LEARNING_RATE = env_float("DRUGBOT_LR", 2e-4)
LORA_R = env_int("DRUGBOT_LORA_R", 16)
LORA_ALPHA = env_int("DRUGBOT_LORA_ALPHA", 32)
SEED = env_int("DRUGBOT_SEED", 42)
# The pinned qa_pairs_2024_v2.jsonl prompts ALREADY embed SYSTEM_PROMPT (they are the full
# serving user turn), so default to verbatim. Set DRUGBOT_PREPEND_SYSTEM=1 only for legacy
# pairs (e.g. combined_pairs.jsonl) whose prompt lacks the persona -- doubling it would
# desync training from serving.
PREPEND_SYSTEM = env_flag("DRUGBOT_PREPEND_SYSTEM", False)

# The text-decoder projections. An explicit list (not "all-linear") keeps LoRA off any vision
# or audio tower this multimodal checkpoint may expose, so trainable params stay small.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def assert_model_loadable(model_id: str) -> None:
    try:
        AutoConfig.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001 -- any config-resolution failure is a hard stop
        raise SystemExit(
            f"Cannot load the base model config for '{model_id}':\n    {exc}\n\n"
            "If this is an 'unrecognized architecture' / 'model type ... not recognized' error, "
            "the installed transformers is too old for this checkpoint:\n"
            "  pip install -U transformers accelerate   (then verify trl/peft still import)\n"
            "and load via the class the model card names. Keep RAG_BASE_MODEL identical when "
            "serving, or the adapter will not attach."
        ) from exc


def require_bitsandbytes_8bit() -> None:

    import importlib

    # (1) strip namespace-package shadows: a `bitsandbytes` dir with no __init__.py directly
    # inside it. A real install has __init__.py, so it is never matched here.
    shadows = []
    for entry in list(sys.path):
        cand = os.path.join(entry or ".", "bitsandbytes")
        if os.path.isdir(cand) and not os.path.exists(os.path.join(cand, "__init__.py")):
            shadows.append(entry)
    if shadows:
        for entry in shadows:
            while entry in sys.path:
                sys.path.remove(entry)
        sys.modules.pop("bitsandbytes", None)
        importlib.invalidate_caches()
        LOG.warning("ignored a shadowing 'bitsandbytes' directory on sys.path (%s); it is a "
                    "source tree, not an installed package", ", ".join(repr(s) for s in shadows))

    def fail(reason: str) -> None:
        raise SystemExit(
            f"8-bit training requires a working bitsandbytes CUDA kernel, but {reason}.\n"
            "Build the CUDA backend for this GPU/CUDA (RTX 5090 = sm_120), then it is picked up\n"
            "automatically by the existing editable install:\n"
            "    cd <the bitsandbytes source checkout>\n"
            "    cmake -DCOMPUTE_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=120 -S . -B build\n"
            "    cmake --build build --config Release -j\n"
            "    pip install -e .\n"
            "Verify:  python -c \"import bitsandbytes as b; print(type(b.cextension.lib).__name__)\"\n"
            "(anything other than 'ErrorHandlerMockBNBNativeLibrary' means the kernel loaded)."
        )

    try:
        import bitsandbytes as bnb
    except Exception as exc:  # noqa: BLE001
        fail(f"it could not be imported ({exc})")
    if getattr(bnb, "__file__", None) is None:
        fail("it imports as an empty namespace package (a directory shadow, not a real install)")
    # bitsandbytes installs a mock library object when its native binary fails to load; its
    # class name is the reliable signal that the CUDA kernel is absent.
    lib = getattr(getattr(bnb, "cextension", None), "lib", None)
    if lib is None or type(lib).__name__ == "ErrorHandlerMockBNBNativeLibrary":
        fail("its native CUDA library was never compiled (libbitsandbytes_cuda*.so is missing)")


def build_quant_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_8bit=True)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model():
    dtype = resolve_dtype()  # int8 quantizes the weights; compute (and LoRA) stays bf16
    load_kwargs = dict(dtype=dtype, device_map={"": 0}, quantization_config=build_quant_config())

    # flash_attention_2 is the throughput path and is required for correct padding-free packing;
    # fall back to sdpa if the arch/build rejects it so the run still proceeds (packing off).
    flash_ok = True
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, attn_implementation="flash_attention_2", **load_kwargs
        )
    except (ImportError, ValueError, RuntimeError) as exc:
        LOG.warning("flash_attention_2 unavailable (%s); using sdpa and disabling packing", exc)
        flash_ok = False
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, attn_implementation="sdpa", **load_kwargs
        )

    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, flash_ok


def assistant_turn_suffix(tokenizer) -> str:

    placeholder = "ASSISTANT_CONTENT"  # unlikely to appear in real data
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": placeholder}],
        add_generation_prompt=False, tokenize=False,
    )
    return rendered.split(placeholder)[-1]


def encode_example(sample: dict, tokenizer, suffix: str, max_len: int) -> dict:
  
    # v2 pairs already embed SYSTEM_PROMPT (full serving user turn); use verbatim unless a
    # legacy dataset needs the persona prepended.
    user = f"{SYSTEM_PROMPT}\n\n{sample['prompt'].strip()}" if PREPEND_SYSTEM else sample["prompt"].strip()
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
    )["input_ids"]
    answer_ids = tokenizer(sample["response"].strip() + suffix, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def build_dataset(tokenizer):
    if not os.path.exists(INSTRUCTIONS):
        raise SystemExit(
            f"Training data not found: {INSTRUCTIONS}\n"
            "Build it first with the merge pipeline (see README): stage 1 dedup_combine_jsonl.py "
            "then stage 2 combine_drug_jsonl.py, which writes data/combined_pairs.jsonl. "
            "Override the path with DRUGBOT_PAIRS."
        )
    raw = load_dataset("json", data_files=INSTRUCTIONS, split="train")
    cols = raw.column_names
    if "prompt" not in cols or "response" not in cols:
        raise SystemExit(
            f"{INSTRUCTIONS} must have 'prompt' and 'response' fields; found {cols}. "
            "This file is the combine_drug_jsonl.py pairs output."
        )
    # Shuffle BEFORE any row cap so a cap does not train on a single source/era slice.
    raw = raw.shuffle(seed=SEED)
    if TRAIN_ROWS and TRAIN_ROWS < raw.num_rows:
        raw = raw.select(range(TRAIN_ROWS))
    suffix = assistant_turn_suffix(tokenizer)
    # Give the map an explicit fingerprint. Otherwise datasets tries to dill-pickle the whole
    # transform -- including this tokenizer -- to build a cache key, and pickling this custom
    # tokenizer recurses deeply enough to overflow the C stack and SIGSEGV (intermittently).
    # A stable fingerprint over the inputs that actually change the output both avoids that
    # pickle AND enables on-disk caching, so the tokenized set is reused across runs. Bump
    # ENCODE_VERSION whenever encode_example / the prompt formatting changes.
    ENCODE_VERSION = ("v2-prepend" if PREPEND_SYSTEM else "v2-verbatim")
    fp_src = f"{MODEL_ID}|{INSTRUCTIONS}|{SEQ_LEN}|{TRAIN_ROWS}|{SEED}|{suffix}|{ENCODE_VERSION}"
    fingerprint = hashlib.md5(fp_src.encode("utf-8")).hexdigest()
    # Run the map IN-PROCESS (num_proc=None), never in a worker pool. In datasets 5.0.0, map()
    # spawns mp.Pool(num_proc) for ANY num_proc >= 1 -- including num_proc=1 -- and ships the task
    # to the worker by dill-pickling fn_kwargs, i.e. this fast (Rust) tokenizer. That pickle
    # recurses deep enough to overflow the C stack and SIGSEGV (intermittently: it may survive one
    # run and crash the next, which is exactly the failure we hit). num_proc=None instead takes the
    # in-process _map_single path, which calls encode_example directly with no IPC pickle. The
    # tokenizer is internally multithreaded, so one process is both safe and fast. (new_fingerprint
    # above already prevents the separate cache-key pickle of this same tokenizer.)
    ds = raw.map(
        encode_example,
        fn_kwargs={"tokenizer": tokenizer, "suffix": suffix, "max_len": SEQ_LEN},
        remove_columns=raw.column_names,
        num_proc=None,
        desc="Tokenizing + labeling",
        new_fingerprint=fingerprint,
    )
    if not EVAL_ENABLED:
        # No eval: keep every row for training (no 5% holdout) and skip the split cost.
        return ds, None
    split = ds.train_test_split(test_size=0.05, seed=SEED)
    eval_ds = split["test"]
    if EVAL_ROWS and EVAL_ROWS < eval_ds.num_rows:
        eval_ds = eval_ds.select(range(EVAL_ROWS))
    return split["train"], eval_ds


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )


def build_sft_config(flash_ok: bool):
    from trl import SFTConfig

    dtype = resolve_dtype()
    # Packing needs flash-attn for correct cross-example masking (padding-free); without it,
    # keep packing off so loss cannot bleed between packed examples.
    packing = flash_ok
    # "8-bit" is the FROZEN BASE weights (loaded int8). The optimizer only updates the small
    # LoRA adapters, so it does not need bitsandbytes' paged 8-bit AdamW -- and that paged
    # optimizer (CUDA managed memory + prefetch) is a prime suspect for the interpreter
    # segfaults on this Blackwell / CUDA-13.2 / bnb-dev stack. Plain fused AdamW; base stays int8.
    optim = "adamw_torch_fused"

    kwargs = dict(
        output_dir=OUTPUT_DIR,
        max_length=SEQ_LEN,
        packing=packing,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(2, BATCH_SIZE),
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=GRAD_CKPT,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=optim,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="steps" if EVAL_ENABLED else "no",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        tf32=True,
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=DATALOADER_WORKERS > 0,
        use_liger_kernel=liger_available(),
        report_to="tensorboard",
    )
    if MAX_STEPS > 0:
        kwargs["max_steps"] = MAX_STEPS
    else:
        kwargs["num_train_epochs"] = EPOCHS
    return SFTConfig(**kwargs)


def liger_available() -> bool:
    if not env_flag("DRUGBOT_LIGER", True):
        return False
    try:
        import liger_kernel  # noqa: F401
        return True
    except ImportError:
        LOG.warning("liger-kernel not installed; skipping fused cross-entropy (higher VRAM). "
                    "Install it for the biggest memory win:  pip install liger-kernel")
        return False


def save_metrics(trainer) -> None:
    """Write loss curves + a per-eval table, joining train/eval logs on 'step' defensively."""
    import pandas as pd

    hist = pd.DataFrame(trainer.state.log_history)
    hist.to_csv("gemma_lora_log_history.csv", index=False)
    if "step" not in hist.columns:
        LOG.warning("no step column in log history; skipping metric plots")
        return

    train_log = hist[hist.get("loss").notna()] if "loss" in hist.columns else hist.iloc[0:0]
    eval_log = hist[hist.get("eval_loss").notna()] if "eval_loss" in hist.columns else hist.iloc[0:0]

    if not eval_log.empty:
        keep = ["step", "eval_loss", "eval_mean_token_accuracy", "eval_entropy"]
        table = eval_log[[c for c in keep if c in eval_log.columns]].copy()
        table.to_csv("gemma_lora_metrics.csv", index=False)
        LOG.info("per-eval metrics:\n%s", table.to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        if not train_log.empty:
            plt.plot(train_log["step"], train_log["loss"], label="training loss")
        if not eval_log.empty:
            plt.plot(eval_log["step"], eval_log["eval_loss"], marker="o", label="validation loss")
        plt.xlabel("step")
        plt.ylabel("loss (cross entropy)")
        plt.title("Gemma LoRA training loss")
        plt.legend()
        plt.savefig("Gemma_Lora_training_loss.png")
        LOG.info("saved Gemma_Lora_training_loss.png")
    except Exception as exc:  # noqa: BLE001 -- plotting must never fail a finished run
        LOG.warning("could not render loss plot: %s", exc)


def main() -> None:
    LOG.info("base model: %s (override with RAG_BASE_MODEL)", MODEL_ID)

    # IMPORTANT ordering: tokenize the whole dataset BEFORE touching CUDA or importing
    # bitsandbytes. A fast (Rust) tokenizer map segfaults once a CUDA context is live in the
    # process, so all fail-fast checks that do not need CUDA (data file, model config) run
    # first, then tokenization, and only then do we initialize CUDA / the 8-bit kernel / model.
    assert_model_loadable(MODEL_ID)
    tokenizer = load_tokenizer()
    train_ds, eval_ds = build_dataset(tokenizer)
    LOG.info("train rows: %d  eval rows: %s", train_ds.num_rows,
             eval_ds.num_rows if eval_ds is not None else "disabled (training on all rows)")

    # From here on CUDA gets initialized. Do the GPU + 8-bit-kernel preflight now (strips any
    # bitsandbytes import shadow and requires a live CUDA kernel), then load the model.
    if not torch.cuda.is_available():
        raise SystemExit("torch cannot see a GPU; QLoRA training needs CUDA.")
    LOG.info("device: %s", torch.cuda.get_device_name(0))
    require_bitsandbytes_8bit()

    model, flash_ok = build_model()
    LOG.info("loaded model: 8-bit=True flash_attn=%s dtype=%s", flash_ok, model.dtype)

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=build_sft_config(flash_ok),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=build_peft_config(),
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Resume is OPT-IN (DRUGBOT_RESUME=1), never automatic. Auto-resuming on every rerun is a
    # footgun: if OUTPUT_DIR holds a checkpoint from a prior run with a different effective batch
    # size, its global_step can already exceed THIS run's total step count, so the trainer runs
    # ZERO steps and exits while still printing "saved adapter" as if it had trained. Default off
    # means a rerun trains fresh and predictably; opt in only to continue an interrupted run of
    # the SAME config (crash insurance on this bleeding-edge GPU stack).
    from transformers.trainer_utils import get_last_checkpoint
    resume = None
    if env_flag("DRUGBOT_RESUME", False):
        resume = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
        if resume:
            LOG.info("DRUGBOT_RESUME=1: resuming from checkpoint %s", resume)
        else:
            LOG.info("DRUGBOT_RESUME=1 but no checkpoint in %s; training from step 0", OUTPUT_DIR)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()
    LOG.info("saved adapter to %s", OUTPUT_DIR)
    if torch.cuda.is_available():
        LOG.info("peak VRAM: %.1f GB", torch.cuda.max_memory_allocated() / 1e9)

    save_metrics(trainer)


if __name__ == "__main__":
    main()
