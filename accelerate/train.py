import json
import math
import torch

from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader, random_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = "/gemini/code/model"      # 本地 Qwen3-8B-Instruct 模型目录
DATA_PATH = "/gemini/code/train.json"  # 你的训练数据（json 或 jsonl）
MAX_LENGTH = 2048
TRAIN_BATCH_SIZE = 1                   # Qwen3-8B + FSDP 推荐从 1 开始
VALID_BATCH_SIZE = 1
LR = 2e-5
EPOCHS = 3
LOG_STEP = 10


def load_raw_data(data_path):
    """
    支持:
    1. JSON 数组文件: [ {...}, {...} ]
    2. JSONL 文件: 每行一个 JSON
    只使用 instruction 和 output 字段
    """
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if text.startswith("["):
        data = json.loads(text)
    else:
        data = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                data.append(json.loads(line))

    raw_data = []
    for item in data:
        if "instruction" not in item or "output" not in item:
            continue

        raw_data.append({
            "instruction": item["instruction"],
            "output": item["output"]
        })

    return raw_data


def prepare_dataloader():
    # ==========================================
    # 1. 读取原始数据
    # ==========================================
    raw_data = load_raw_data(DATA_PATH)
    dataset = Dataset.from_list(raw_data)

    # ==========================================
    # 2. tokenizer
    # ==========================================
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==========================================
    # 3. 单条样本预处理（SFT）
    # ==========================================
    def preprocess(example):
        # 使用 Qwen 聊天模板
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]

        # 完整对话文本
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # 仅 user 部分（用于确定需要屏蔽的位置）
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True
        )

        # 编码
        full_ids = tokenizer.encode(
            full_text,
            add_special_tokens=False
        )

        prompt_ids = tokenizer.encode(
            prompt_text,
            add_special_tokens=False
        )

        # labels: prompt 部分不参与 loss
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 截断
        input_ids = full_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

        # attention_mask
        attention_mask = [1] * len(input_ids)

        # padding
        pad_len = MAX_LENGTH - len(input_ids)

        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # ==========================================
    # 4. 提前 tokenization
    # ==========================================
    tokenized_dataset = dataset.map(
        preprocess,
        remove_columns=dataset.column_names
    )

    # ==========================================
    # 5. 划分训练集/验证集
    # ==========================================
    total_size = len(tokenized_dataset)
    train_size = max(1, int(total_size * 0.9))
    valid_size = total_size - train_size

    if valid_size == 0:
        train_size = total_size - 1
        valid_size = 1

    trainset, validset = random_split(
        tokenized_dataset,
        [train_size, valid_size],
        generator=torch.Generator().manual_seed(42)
    )

    # ==========================================
    # 6. collate_fn：仅转 tensor
    # ==========================================
    def collate_func(batch):
        return {
            "input_ids": torch.tensor(
                [item["input_ids"] for item in batch],
                dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [item["attention_mask"] for item in batch],
                dtype=torch.long
            ),
            "labels": torch.tensor(
                [item["labels"] for item in batch],
                dtype=torch.long
            )
        }

    # ==========================================
    # 7. DataLoader
    # ==========================================
    trainloader = DataLoader(
        trainset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_func
    )

    validloader = DataLoader(
        validset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_func
    )

    return trainloader, validloader


def prepare_model_and_optimizer():
    # Qwen3-8B Causal LM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR
    )

    return model, optimizer


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for batch in validloader:
            outputs = model(**batch)
            loss = outputs.loss

            # 多卡聚合
            loss = accelerator.gather_for_metrics(loss.detach())
            total_loss += loss.mean().item()

    avg_loss = total_loss / len(validloader)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    return avg_loss, ppl


def train(
    model,
    optimizer,
    trainloader,
    validloader,
    accelerator: Accelerator,
    epoch=EPOCHS,
    log_step=LOG_STEP
):
    global_step = 0

    for ep in range(epoch):
        model.train()

        for batch in trainloader:
            optimizer.zero_grad()

            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()

            if global_step % log_step == 0:
                reduced_loss = accelerator.reduce(loss.detach(), "mean")
                accelerator.print(
                    f"ep: {ep}, "
                    f"global_step: {global_step}, "
                    f"loss: {reduced_loss.item():.6f}"
                )

            global_step += 1

        eval_loss, ppl = evaluate(model, validloader, accelerator)

        accelerator.print(
            f"ep: {ep}, "
            f"eval_loss: {eval_loss:.6f}, "
            f"ppl: {ppl:.4f}"
        )


def main():
    # 使用 accelerate config 配置 FSDP 后，这里无需额外参数
    accelerator = Accelerator()

    trainloader, validloader = prepare_dataloader()

    model, optimizer = prepare_model_and_optimizer()

    model, optimizer, trainloader, validloader = accelerator.prepare(
        model,
        optimizer,
        trainloader,
        validloader
    )

    train(
        model,
        optimizer,
        trainloader,
        validloader,
        accelerator
    )


if __name__ == "__main__":
    main()