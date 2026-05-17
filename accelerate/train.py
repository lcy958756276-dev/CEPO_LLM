import json
import math
import torch
#accelerate launch --config_file fsdp_config.yaml train.py
import json
import math
import torch

from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig, get_peft_model, TaskType


MODEL_PATH = "/root/CEPO_LLM/Qwen3-8B"
DATA_PATH = "/root/CEPO_LLM/data/fsdp_data.json"

MAX_LENGTH = 1024   # ⭐建议先1024，2048很容易炸
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1

LR = 2e-4
EPOCHS = 3
LOG_STEP = 10


# =========================
# 数据
# =========================
def load_raw_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if text.startswith("["):
        data = json.loads(text)
    else:
        data = [json.loads(l) for l in text.splitlines() if l.strip()]

    return [
        {"instruction": x["instruction"], "output": x["output"]}
        for x in data
        if "instruction" in x and "output" in x
    ]


def prepare_dataloader():
    raw = load_raw_data(DATA_PATH)
    dataset = Dataset.from_list(raw)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        full_ids = full_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        attention_mask = [1] * len(full_ids)

        pad_len = MAX_LENGTH - len(full_ids)

        full_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    train_size = int(len(dataset) * 0.9)
    valid_size = len(dataset) - train_size

    trainset, validset = random_split(dataset, [train_size, valid_size])

    def collate(batch):
        return {
            "input_ids": torch.tensor([x["input_ids"] for x in batch]),
            "attention_mask": torch.tensor([x["attention_mask"] for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    return (
        DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collate),
        DataLoader(validset, batch_size=VALID_BATCH_SIZE, shuffle=False, collate_fn=collate)
    )


# =========================
# 模型 + LoRA
# =========================
def prepare_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # ===== LoRA =====
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    )

    model = get_peft_model(model, lora_config)
    model = model.to(torch.bfloat16)
    model.print_trainable_parameters()

    return model


# =========================
# eval
# =========================
def evaluate(model, loader, accelerator):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            loss = out.loss
            loss = accelerator.gather_for_metrics(loss)
            total += loss.mean().item()

    avg = total / len(loader)
    return avg, math.exp(avg) if avg < 20 else float("inf")


# =========================
# train
# =========================
def train(model, optimizer, trainloader, validloader, accelerator):

    step = 0

    for ep in range(EPOCHS):
        model.train()

        for batch in trainloader:

            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()

            if step % LOG_STEP == 0:
                accelerator.print(f"ep {ep} step {step} loss {loss.item():.4f}")

            step += 1

        val_loss, ppl = evaluate(model, validloader, accelerator)

        accelerator.print(
            f"[EPOCH {ep}] val_loss={val_loss:.4f} ppl={ppl:.2f}"
        )


# =========================
# main
# =========================
def main():

    accelerator = Accelerator()

    trainloader, validloader = prepare_dataloader()

    model = prepare_model()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    model, optimizer, trainloader, validloader = accelerator.prepare(
        model, optimizer, trainloader, validloader
    )

    train(model, optimizer, trainloader, validloader, accelerator)


if __name__ == "__main__":
    main()