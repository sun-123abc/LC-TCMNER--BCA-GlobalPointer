# -*- coding: utf-8 -*-
"""
Clean train.py for TCM NER comparison experiments.

当前版本：11 类实体体系，已删除“既往病史”。
支持模型：
  已有 5 个：
    1) bert_crf              -> model.baseline.bert_crf
    2) bert_bilstm_crf       -> model.baseline.bert_bilstm_crf
    3) global_pointer        -> model.baseline.GlobalPointer
    4) bpbic                 -> model.migrated.bpbic
    5) hreb_crf              -> model.migrated.hreb_crf

  新增 5 个：
    6) idcnn_crf             -> model.baseline.idcnn_crf
    7) transformer_crf       -> model.baseline.transformer_crf
    8) efficient_global_pointer -> model.baseline.efficient_global_pointer
    9) tplinker_ner          -> model.baseline.tplinker_ner
   10) biaffine_ner          -> model.baseline.biaffine_ner

设计原则：
  - 固定同一套数据读取、训练、验证、测试、评价逻辑；
  - 默认不做自动阈值搜索，span 模型使用 threshold=0.0；
  - 每个模型单独保存到 checkpoints/{model_name}/；
  - 输出报告到 outputs/{model_name}_train_report.txt；
  - 自动检查模型模块、参数分组和 checkpoint，避免跑错模型。
"""

import os
import sys
import json
import time
import random
import argparse
import importlib
from collections import Counter
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


# =========================================================
# 1. 路径设置
# =========================================================
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "dataset_final")
CHECKPOINT_ROOT = os.path.join(ROOT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_final.txt")
DEV_PATH = os.path.join(DATA_DIR, "dev_final.txt")
TEST_PATH = os.path.join(DATA_DIR, "test_final.txt")


# =========================================================
# 2. 本地预训练模型路径
# =========================================================
LOCAL_PRETRAINED_DIR = os.path.join(
    ROOT_DIR,
    "pretrained_models",
    "chinese-bert-wwm-ext"
)


# =========================================================
# 3. 实体类别：11 类，已删除“既往病史”
# =========================================================
ENTITY_TYPES = [
    "患者信息",
    "时间",
    "西医病名",
    "中医病名",
    "症状",
    "检查",
    "检查结果",
    "治法",
    "证型",
    "处方用药",
    "方剂",
]
VALID_ENTITY_TYPES = set(ENTITY_TYPES)


# =========================================================
# 4. 模型注册表
# =========================================================
MODEL_REGISTRY = {
    # 原有核心对比模型
    "bert_crf": "model.baseline.bert_crf",
    "bert_bilstm_crf": "model.baseline.bert_bilstm_crf",
    "global_pointer": "model.baseline.GlobalPointer",
    "bpbic": "model.migrated.bpbic",
    "hreb_crf": "model.migrated.hreb_crf",

    # 已补充对比模型
    "idcnn_crf": "model.baseline.idcnn_crf",
    "efficient_global_pointer": "model.baseline.efficient_global_pointer",
    "tplinker_ner": "model.baseline.tplinker_ner",
    "biaffine_ner": "model.baseline.biaffine_ner",

    # 替换 Transformer-CRF 的基础对比模型
    "bert_softmax": "model.baseline.bert_softmax",

    # proposed：第一个创新点，暂时命名为 model1
    "model1": "model.proposed.model1",
    # proposed：第二个创新点，基于 model1 继续增强
    "model2": "model.proposed.model2",
    "model3": "model.proposed.BCA-GlobalPointer",
    "bca_global_pointer": "model.proposed.bca_global_pointer",
}


# =========================================================
# 5. 通用工具
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}小时{m}分{s}秒"
    if m > 0:
        return f"{m}分{s}秒"
    return f"{s}秒"


def check_file_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 不存在：{path}")


def load_model_module(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"未知模型：{model_name}\n"
            f"当前支持模型：{list(MODEL_REGISTRY.keys())}"
        )

    module_path = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)

    if not hasattr(module, "MODEL_TYPE"):
        raise ValueError(f"{module_path} 缺少 MODEL_TYPE，例如 sequence_labeling / span_labeling")

    if not hasattr(module, "build_model"):
        raise ValueError(f"{module_path} 缺少 build_model()")

    return module_path, module.MODEL_TYPE, module.build_model


def build_model_safely(build_model, pretrained_model: str, num_labels: int, args):
    """
    兼容不同模型 build_model 写法。
    优先使用关键字参数；如果旧模型不支持，再尝试位置参数。
    """
    try:
        return build_model(
            pretrained_model_name_or_path=pretrained_model,
            num_labels=num_labels,
            dropout_rate=args.dropout_rate,
        )
    except TypeError:
        return build_model(
            pretrained_model,
            num_labels,
            args.dropout_rate,
        )


def is_backbone_param(name: str) -> bool:
    """
    判断参数是否属于预训练编码器，统一使用 bert_lr。
    兼容 bert / encoder / roberta / backbone 等不同命名。
    非预训练模型的 embedding/encoder 将自动归入 other_lr。
    """
    low = name.lower()
    backbone_keywords = [
        "bert.",
        "encoder.",
        "roberta.",
        "macbert.",
        "backbone.",
        "pretrained_model.",
        "transformer_model.",
    ]
    return any(k in low for k in backbone_keywords)


def print_model_debug_info(model, args, module_path: str):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 100)
    print("[模型身份检查]")
    print(f"model_name: {args.model_name}")
    print(f"module_path: {module_path}")
    print(f"model_class: {model.__class__.__module__}.{model.__class__.__name__}")
    print(f"total_params: {total_params:,}")
    print(f"trainable_params: {trainable_params:,}")

    print("\n[前 30 个参数名]")
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx >= 30:
            break
        print(f"{idx + 1:02d}. {name} {tuple(param.shape)}")
    print("=" * 100)


# =========================================================
# 6. BIO 数据读取与实体转换
# =========================================================
def read_bio_file(path: str) -> List[Tuple[List[str], List[str]]]:
    check_file_exists(path, "BIO 数据文件")

    samples = []
    chars, labels = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                if chars:
                    samples.append((chars, labels))
                    chars, labels = [], []
                continue

            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                # 兼容 tab 或多个空白
                parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                continue

            ch, lab = parts
            chars.append(ch)
            labels.append(lab)

    if chars:
        samples.append((chars, labels))

    return samples


def build_sequence_label_map():
    label_list = ["O"]
    for ent in ENTITY_TYPES:
        label_list.append("B-" + ent)
        label_list.append("I-" + ent)

    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def build_span_label_map():
    label2id = {lab: i for i, lab in enumerate(ENTITY_TYPES)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def bio_to_entities(labels: List[str], valid_entity_types: Optional[Set[str]] = None) -> Set[Tuple[int, int, str]]:
    entities = set()
    start = None
    ent_type = None

    def _add_entity(s, e, t):
        if t is None:
            return
        if valid_entity_types is not None and t not in valid_entity_types:
            return
        entities.add((s, e, t))

    for i, lab in enumerate(labels):
        if lab == "O":
            if start is not None:
                _add_entity(start, i - 1, ent_type)
                start = None
                ent_type = None
            continue

        if lab.startswith("B-"):
            if start is not None:
                _add_entity(start, i - 1, ent_type)
            start = i
            ent_type = lab[2:]

        elif lab.startswith("I-"):
            cur_type = lab[2:]
            if start is None:
                start = i
                ent_type = cur_type
            elif cur_type != ent_type:
                _add_entity(start, i - 1, ent_type)
                start = i
                ent_type = cur_type
        else:
            if start is not None:
                _add_entity(start, i - 1, ent_type)
                start = None
                ent_type = None

    if start is not None:
        _add_entity(start, len(labels) - 1, ent_type)

    return entities


def sequence_ids_to_entities(pred_ids: List[int], id2label: Dict[int, str]) -> Set[Tuple[int, int, str]]:
    labs = [id2label.get(int(x), "O") for x in pred_ids]
    return bio_to_entities(labs, valid_entity_types=VALID_ENTITY_TYPES)


def count_total_entities(samples: List[Tuple[List[str], List[str]]]) -> int:
    total = 0
    for _, labels in samples:
        total += len(bio_to_entities(labels, valid_entity_types=VALID_ENTITY_TYPES))
    return total


# =========================================================
# 7. Dataset
# =========================================================
class NERDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[List[str], List[str]]],
        tokenizer,
        model_type: str,
        label2id: Dict[str, int],
        max_len: int,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.label2id = label2id
        self.max_len = max_len

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        self.unk_id = tokenizer.unk_token_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        chars, labels = self.samples[idx]

        chars = chars[: self.max_len - 2]
        labels = labels[: self.max_len - 2]

        input_ids = [self.cls_id]
        attention_mask = [1]

        for ch in chars:
            token_id = self.tokenizer.convert_tokens_to_ids(ch)
            if token_id is None:
                token_id = self.unk_id
            input_ids.append(token_id)
            attention_mask.append(1)

        input_ids.append(self.sep_id)
        attention_mask.append(1)

        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [self.pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        gold_entities = bio_to_entities(labels, valid_entity_types=VALID_ENTITY_TYPES)

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "gold_entities": gold_entities,
            "raw_length": len(chars),
            "sample_index": idx,
        }

        if self.model_type == "sequence_labeling":
            label_ids = [self.label2id["O"]]
            for lab in labels:
                label_ids.append(self.label2id.get(lab, self.label2id["O"]))
            label_ids.append(self.label2id["O"])
            label_ids = label_ids + [self.label2id["O"]] * pad_len
            item["labels"] = torch.tensor(label_ids, dtype=torch.long)

        elif self.model_type == "span_labeling":
            span_labels = torch.zeros(
                len(self.label2id),
                self.max_len,
                self.max_len,
                dtype=torch.float,
            )

            for start, end, ent_type in gold_entities:
                if ent_type not in self.label2id:
                    continue
                ent_id = self.label2id[ent_type]
                start_pos = start + 1
                end_pos = end + 1
                if 0 <= start_pos < self.max_len and 0 <= end_pos < self.max_len:
                    span_labels[ent_id, start_pos, end_pos] = 1.0

            item["labels"] = span_labels

        else:
            raise ValueError(f"不支持的 MODEL_TYPE: {self.model_type}")

        return item


def collate_fn(batch):
    output = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        output[key] = torch.stack([item[key] for item in batch], dim=0)
    output["gold_entities"] = [item["gold_entities"] for item in batch]
    output["raw_length"] = [item["raw_length"] for item in batch]
    output["sample_index"] = [item["sample_index"] for item in batch]
    return output


# =========================================================
# 8. 评价函数
# =========================================================
def compute_metrics(all_pred_entities, all_gold_entities):
    total_pred = 0
    total_gold = 0
    total_correct = 0

    pred_counter = Counter()
    gold_counter = Counter()
    correct_counter = Counter()

    for pred_entities, gold_entities in zip(all_pred_entities, all_gold_entities):
        pred_entities = {e for e in set(pred_entities) if e[2] in VALID_ENTITY_TYPES}
        gold_entities = {e for e in set(gold_entities) if e[2] in VALID_ENTITY_TYPES}
        correct_entities = pred_entities & gold_entities

        total_pred += len(pred_entities)
        total_gold += len(gold_entities)
        total_correct += len(correct_entities)

        for ent in pred_entities:
            pred_counter[ent[2]] += 1
        for ent in gold_entities:
            gold_counter[ent[2]] += 1
        for ent in correct_entities:
            correct_counter[ent[2]] += 1

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    per_label = {}
    for label in ENTITY_TYPES:
        p = correct_counter[label] / pred_counter[label] if pred_counter[label] > 0 else 0.0
        r = correct_counter[label] / gold_counter[label] if gold_counter[label] > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        per_label[label] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "gold": gold_counter[label],
            "pred": pred_counter[label],
            "correct": correct_counter[label],
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_gold": total_gold,
        "total_pred": total_pred,
        "total_correct": total_correct,
        "per_label": per_label,
    }


def decode_span_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    id2label: Dict[int, str],
    threshold: float = 0.0,
    max_span_len: int = 50,
):
    logits = logits.detach()
    attention_mask = attention_mask.detach()
    batch_size, ent_type_size, seq_len, _ = logits.size()
    all_entities = []

    for b in range(batch_size):
        valid_len = int(attention_mask[b].sum().item())
        cur_logits = logits[b, :, 1: valid_len - 1, 1: valid_len - 1]
        cur_len = cur_logits.size(-1)

        if cur_len <= 0:
            all_entities.append(set())
            continue

        if max_span_len is not None and max_span_len > 0:
            idx = torch.arange(cur_len, device=logits.device)
            start_idx = idx.view(-1, 1)
            end_idx = idx.view(1, -1)
            length_mask = (end_idx >= start_idx) & ((end_idx - start_idx + 1) <= max_span_len)
            cur_logits = cur_logits.masked_fill(~length_mask.unsqueeze(0), -1e12)

        positions = torch.nonzero(cur_logits > float(threshold), as_tuple=False)
        entities = set()
        for ent_id, start, end in positions.tolist():
            ent_label = id2label[int(ent_id)]
            if ent_label in VALID_ENTITY_TYPES:
                entities.add((int(start), int(end), ent_label))
        all_entities.append(entities)

    return all_entities


def get_output_value(outputs: Any, key: str, default=None):
    if isinstance(outputs, dict):
        return outputs.get(key, default)
    if hasattr(outputs, key):
        return getattr(outputs, key)
    return default


def evaluate(
    model,
    dataloader,
    model_type: str,
    id2label: Dict[int, str],
    device,
    span_threshold: float = 0.0,
    max_span_len: int = 50,
    desc: str = "Evaluating",
):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_pred_entities = []
    all_gold_entities = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False, ncols=120):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = get_output_value(outputs, "loss", None)
            logits = get_output_value(outputs, "logits", None)
            pred_ids_batch = get_output_value(outputs, "pred_ids", None)

            if loss is not None:
                total_loss += float(loss.item())
                total_steps += 1

            gold_entities = batch["gold_entities"]

            if model_type == "sequence_labeling":
                attention_np = attention_mask.detach().cpu().numpy()
                if pred_ids_batch is None:
                    if logits is None:
                        raise ValueError("sequence_labeling 模型输出中既没有 pred_ids，也没有 logits")
                    pred_ids_batch = torch.argmax(logits, dim=-1).detach().cpu().tolist()

                pred_entities_batch = []
                for i, pred_ids in enumerate(pred_ids_batch):
                    seq_len = int(attention_np[i].sum())
                    pred_ids = pred_ids[1: seq_len - 1]
                    pred_entities = sequence_ids_to_entities(pred_ids, id2label)
                    pred_entities_batch.append(pred_entities)

            elif model_type == "span_labeling":
                if logits is None:
                    raise ValueError("span_labeling 模型输出中没有 logits")
                pred_entities_batch = decode_span_logits(
                    logits=logits,
                    attention_mask=attention_mask,
                    id2label=id2label,
                    threshold=span_threshold,
                    max_span_len=max_span_len,
                )
            else:
                raise ValueError(f"不支持的 MODEL_TYPE: {model_type}")

            all_pred_entities.extend(pred_entities_batch)
            all_gold_entities.extend(gold_entities)

    metrics = compute_metrics(all_pred_entities, all_gold_entities)
    metrics["loss"] = total_loss / total_steps if total_steps > 0 else 0.0
    return metrics


def print_metrics(title: str, metrics: Dict[str, Any]):
    print(f"\n===== {title} =====")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Gold:      {metrics['total_gold']}")
    print(f"Pred:      {metrics['total_pred']}")
    print(f"Correct:   {metrics['total_correct']}")


# =========================================================
# 9. 参数
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert_crf",
        choices=list(MODEL_REGISTRY.keys()),
        help="要训练的模型名称",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=LOCAL_PRETRAINED_DIR,
        help="本地预训练模型路径，默认使用 pretrained_models/chinese-bert-wwm-ext",
    )
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bert_lr", type=float, default=2e-5)
    parser.add_argument("--other_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--span_threshold", type=float, default=0.0)
    parser.add_argument("--max_span_len", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


# =========================================================
# 10. 主流程
# =========================================================
def main():
    args = parse_args()
    set_seed(args.seed)

    check_file_exists(args.pretrained_model, "本地预训练模型目录")
    check_file_exists(TRAIN_PATH, "训练集")
    check_file_exists(DEV_PATH, "验证集")
    check_file_exists(TEST_PATH, "测试集")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module_path, model_type, build_model = load_model_module(args.model_name)

    checkpoint_dir = os.path.join(CHECKPOINT_ROOT, args.model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    label_map_path = os.path.join(checkpoint_dir, "label_map.json")
    report_path = os.path.join(OUTPUT_DIR, f"{args.model_name}_train_report.txt")

    print("=" * 100)
    print("干净版 NER 对比实验训练脚本")
    print(f"项目路径: {ROOT_DIR}")
    print(f"当前数据目录: {DATA_DIR}")
    print(f"训练集路径: {TRAIN_PATH}")
    print(f"验证集路径: {DEV_PATH}")
    print(f"测试集路径: {TEST_PATH}")
    print(f"模型名称: {args.model_name}")
    print(f"模型模块: {module_path}")
    print(f"模型类型: {model_type}")
    print(f"预训练模型: {args.pretrained_model}")
    print(f"设备: {device}")
    print(f"seed: {args.seed}")
    print("=" * 100)

    train_samples = read_bio_file(TRAIN_PATH)
    dev_samples = read_bio_file(DEV_PATH)
    test_samples = read_bio_file(TEST_PATH)

    print(f"训练集样本数: {len(train_samples)}")
    print(f"验证集样本数: {len(dev_samples)}")
    print(f"测试集样本数: {len(test_samples)}")
    print(f"训练集实体数 Gold: {count_total_entities(train_samples)}")
    print(f"验证集实体数 Gold: {count_total_entities(dev_samples)}")
    print(f"测试集实体数 Gold: {count_total_entities(test_samples)}")

    if model_type == "sequence_labeling":
        label2id, id2label = build_sequence_label_map()
    elif model_type == "span_labeling":
        label2id, id2label = build_span_label_map()
    else:
        raise ValueError(f"不支持的 MODEL_TYPE: {model_type}")

    num_labels = len(label2id)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "model_type": model_type,
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "entity_types": ENTITY_TYPES,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"标签数量: {num_labels}")
    print(f"标签映射保存到: {label_map_path}")

    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    train_dataset = NERDataset(train_samples, tokenizer, model_type, label2id, args.max_len)
    dev_dataset = NERDataset(dev_samples, tokenizer, model_type, label2id, args.max_len)
    test_dataset = NERDataset(test_samples, tokenizer, model_type, label2id, args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)

    print("\n加载模型...")
    model = build_model_safely(build_model=build_model, pretrained_model=args.pretrained_model, num_labels=num_labels, args=args)
    model.to(device)
    print_model_debug_info(model, args, module_path)

    bert_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_backbone_param(name):
            bert_params.append((name, param))
        else:
            other_params.append((name, param))

    print("=" * 100)
    print("[参数分组检查]")
    print(f"BERT/Encoder 参数量: {sum(p.numel() for _, p in bert_params):,}")
    print(f"Other 参数量:        {sum(p.numel() for _, p in other_params):,}")
    print(f"bert_lr:  {args.bert_lr}")
    print(f"other_lr: {args.other_lr}")
    print("=" * 100)

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in bert_params if not any(nd in n for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"])],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr,
        },
        {
            "params": [p for n, p in bert_params if any(nd in n for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"])],
            "weight_decay": 0.0,
            "lr": args.bert_lr,
        },
        {
            "params": [p for _, p in other_params],
            "weight_decay": 0.0,
            "lr": args.other_lr,
        },
    ]
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    best_dev_f1 = -1.0
    best_epoch = 0
    no_improve_epochs = 0
    train_logs = []

    print("\n开始训练...")
    print(f"Epochs: {args.epochs}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print("-" * 100)

    total_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        step_count = 0

        progress_bar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Training Epoch {epoch}/{args.epochs}", ncols=120)
        for step, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = get_output_value(outputs, "loss", None)
            if loss is None:
                raise ValueError("模型 forward 输出中没有 loss，请检查模型文件。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += float(loss.item())
            step_count += 1
            avg_loss = total_train_loss / step_count
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({"step": f"{step}/{len(train_loader)}", "loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

        avg_train_loss = total_train_loss / max(step_count, 1)
        train_time = time.time() - epoch_start
        dev_metrics = evaluate(model=model, dataloader=dev_loader, model_type=model_type, id2label=id2label, device=device, span_threshold=args.span_threshold, max_span_len=args.max_span_len, desc="Evaluating dev")
        epoch_time = time.time() - epoch_start

        log_line = (
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"dev_loss={dev_metrics['loss']:.4f} | "
            f"dev_P={dev_metrics['precision']:.4f} | "
            f"dev_R={dev_metrics['recall']:.4f} | "
            f"dev_F1={dev_metrics['f1']:.4f} | "
            f"train_time={format_time(train_time)} | "
            f"epoch_time={format_time(epoch_time)}"
        )
        train_logs.append(log_line)
        print("\n" + log_line)

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(
                {
                    "model_name": args.model_name,
                    "module_path": module_path,
                    "model_type": model_type,
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "entity_types": ENTITY_TYPES,
                    "pretrained_model": args.pretrained_model,
                    "max_len": args.max_len,
                    "span_threshold": args.span_threshold,
                    "max_span_len": args.max_span_len,
                    "best_dev_f1": best_dev_f1,
                    "best_epoch": best_epoch,
                    "seed": args.seed,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"验证集 F1 提升，保存最佳模型：{best_model_path}")
            print(f"Best Dev F1: {best_dev_f1:.4f} at Epoch {best_epoch}")
        else:
            no_improve_epochs += 1
            print(f"验证集 F1 未提升：{no_improve_epochs}/{args.patience}")
            print(f"当前 Best Dev F1: {best_dev_f1:.4f} at Epoch {best_epoch}")

        print("-" * 100)
        if no_improve_epochs >= args.patience:
            print(f"\n触发早停：连续 {args.patience} 个 epoch 未提升。")
            break

    total_train_time = time.time() - total_start_time

    print("\n加载最佳模型并测试...")
    checkpoint = torch.load(best_model_path, map_location=device)
    ckpt_model_name = checkpoint.get("model_name", None)
    if ckpt_model_name != args.model_name:
        raise ValueError(f"checkpoint 模型名不匹配：checkpoint={ckpt_model_name}, 当前={args.model_name}")
    model.load_state_dict(checkpoint["model_state_dict"])

    print("=" * 100)
    print("[Checkpoint 检查]")
    print(f"checkpoint path: {best_model_path}")
    print(f"checkpoint model_name: {checkpoint.get('model_name')}")
    print(f"checkpoint module_path: {checkpoint.get('module_path')}")
    print(f"best_epoch: {checkpoint.get('best_epoch')}")
    print(f"best_dev_f1: {checkpoint.get('best_dev_f1'):.4f}")
    print("=" * 100)

    test_metrics = evaluate(model=model, dataloader=test_loader, model_type=model_type, id2label=id2label, device=device, span_threshold=args.span_threshold, max_span_len=args.max_span_len, desc="Testing")
    print_metrics("测试集整体结果", test_metrics)

    print("\n===== 测试集分实体类别结果 =====")
    for label, m in test_metrics["per_label"].items():
        print(f"{label}\tP={m['precision']:.4f}\tR={m['recall']:.4f}\tF1={m['f1']:.4f}\tGold={m['gold']}\tPred={m['pred']}\tCorrect={m['correct']}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"===== {args.model_name} 训练报告 =====\n")
        f.write(f"模型模块: {module_path}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"预训练模型: {args.pretrained_model}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"max_len: {args.max_len}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"bert_lr: {args.bert_lr}\n")
        f.write(f"other_lr: {args.other_lr}\n")
        f.write(f"dropout_rate: {args.dropout_rate}\n")
        f.write(f"span_threshold: {args.span_threshold}\n")
        f.write(f"最佳验证集 F1: {best_dev_f1:.4f}\n")
        f.write(f"最佳 Epoch: {best_epoch}\n")
        f.write(f"总训练耗时: {format_time(total_train_time)}\n\n")
        f.write("===== 训练日志 =====\n")
        for line in train_logs:
            f.write(line + "\n")
        f.write("\n===== 测试集整体结果 =====\n")
        f.write(f"Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"F1:        {test_metrics['f1']:.4f}\n")
        f.write(f"Gold:      {test_metrics['total_gold']}\n")
        f.write(f"Pred:      {test_metrics['total_pred']}\n")
        f.write(f"Correct:   {test_metrics['total_correct']}\n")
        f.write("\n===== 测试集分实体类别结果 =====\n")
        for label, m in test_metrics["per_label"].items():
            f.write(f"{label}\tP={m['precision']:.4f}\tR={m['recall']:.4f}\tF1={m['f1']:.4f}\tGold={m['gold']}\tPred={m['pred']}\tCorrect={m['correct']}\n")

    print(f"\n训练报告已保存：{report_path}")


if __name__ == "__main__":
    main()
