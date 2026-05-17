# -*- coding: utf-8 -*-
"""
BCA-GlobalPointer.py

Model3_v14_conservative：回滚到 v11 稳定路线的保守增强版 GlobalPointer
当前作为第三个创新点的临时代码名：model3

本版是在 model3_v8 基础上的保守微调版，仍然严格遵守：
1. 不改 train.py 的训练策略；
2. 不改数据划分；
3. 不改评价方式；
4. 仍以 GlobalPointer 为直接基线；
5. 只在模型结构内部增强。

核心改动：
1. 修正上一版局部卷积模块中的残差叠加过强问题；
2. 引入多尺度局部上下文增强，强化短实体和边界邻域表达；
3. 引入类别相关的轻量 Biaffine span 打分头，与 GlobalPointer 形成互补；
4. 引入边界感知 span 偏置，但使用 tanh 限幅，避免上一版 Recall 提升但 Precision 下降过多；
5. 引入可学习 span length bias，帮助模型学习实体长度分布；
6. 保留 boundary loss，并加入很小权重的 focal span loss，增强难例 span；
7. v14 回滚到当前最好结果 v11 的稳定配置，只做极小幅处方用药补偿，并收紧方剂补偿，避免 v12/v13 中 Pred 暴涨导致 Precision 下滑。

本版严格基于已经锁定的 model1 与 model2：训练阶段优先从 checkpoints/model2/best_model.pt 初始化；验证/测试阶段继续使用冻结的 model1 参考分支。
在 model2 的“证型、处方用药”静态路由基础上，保留小类 Span 残差适配器，并在验证/测试阶段使用更偏向“处方用药/检查结果/方剂”的长度门控召回平衡校准，目标是在保持高 Precision 的同时补回部分 Recall。

适配当前 clean train.py：
    MODEL_TYPE = "span_labeling"
    build_model(pretrained_model_name_or_path, num_labels, dropout_rate)

输出：
    {"loss": loss, "logits": logits}
"""

import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


MODEL_TYPE = "span_labeling"


# =========================================================
# GlobalPointer loss
# =========================================================
def multilabel_categorical_crossentropy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    GlobalPointer 多标签交叉熵。
    y_pred/y_true: [B, C, L, L]
    """
    y_true = y_true.float()
    y_pred = y_pred.float()

    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], -1)
    y_true = y_true.reshape(y_true.shape[0] * y_true.shape[1], -1)

    y_pred = (1.0 - 2.0 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1.0 - y_true) * 1e12

    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


# =========================================================
# Multi-scale local context enhancement
# =========================================================
class MultiScaleLocalContext(nn.Module):
    """
    多尺度局部上下文增强模块。

    说明：
    - 使用 depthwise conv 捕获 3/5/7 字窗口；
    - 使用门控残差融合；
    - 内部 dropout 不直接使用 train.py 的 0.5，而是限制到 0.2，避免新增模块过强随机失活。
    """

    def __init__(self, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        module_dropout = min(float(dropout_rate), 0.20)

        self.dw_conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.dw_conv5 = nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size)
        self.dw_conv7 = nn.Conv1d(hidden_size, hidden_size, kernel_size=7, padding=3, groups=hidden_size)
        self.pw_conv = nn.Conv1d(hidden_size * 3, hidden_size, kernel_size=1)

        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(module_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x_t = x.transpose(1, 2)

        c3 = self.dw_conv3(x_t)
        c5 = self.dw_conv5(x_t)
        c7 = self.dw_conv7(x_t)
        conv = torch.cat([c3, c5, c7], dim=1)
        conv = self.pw_conv(conv).transpose(1, 2)
        conv = F.gelu(conv)
        conv = self.dropout(conv)

        gate = torch.sigmoid(self.gate(torch.cat([residual, conv], dim=-1)))
        out = residual + gate * conv

        if attention_mask is not None:
            out = out * attention_mask.unsqueeze(-1).float()

        return self.layer_norm(out)


# =========================================================
# Model2
# =========================================================
class Model3GlobalPointer(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_labels: int,
        inner_dim: int = 64,
        biaffine_dim: int = 32,
        max_len_for_bias: int = 256,
        dropout_rate: float = 0.1,
        use_rope: bool = True,
        use_local_context: bool = True,
        use_boundary: bool = True,
        use_biaffine: bool = True,
        use_length_bias: bool = True,
        boundary_loss_weight: float = 0.04,
        boundary_logit_weight: float = 0.07,
        biaffine_logit_weight: float = 0.10,
        focal_loss_weight: float = 0.010,
        focal_gamma: float = 2.0,
        use_small_class_refine: bool = True,
        refine_reg_weight: float = 6e-5,
        auto_load_model2: bool = True,
        model2_ckpt_path: Optional[str] = None,
        auto_load_model1_reference: bool = True,
        model1_ckpt_path: Optional[str] = None,
        use_reference_ensemble: bool = True,
        dynamic_blend: float = 0.45,
        dynamic_margin: float = 0.20,
        dynamic_temperature: float = 4.0,
        use_final_eval_calibration: bool = True,
        final_eval_calibration_scale: float = 1.0,
        use_boundary_gated_calibration: bool = True,
        calibration_gate_floor: float = 0.55,
        calibration_gate_strength: float = 0.45,
        freeze_base_model: bool = True,
        is_reference_model: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        hidden_size = self.encoder.config.hidden_size

        self.num_labels = int(num_labels)
        self.inner_dim = int(inner_dim)
        self.biaffine_dim = int(biaffine_dim)
        self.max_len_for_bias = int(max_len_for_bias)
        self.use_rope = bool(use_rope)
        self.use_local_context = bool(use_local_context)
        self.use_boundary = bool(use_boundary)
        self.use_biaffine = bool(use_biaffine)
        self.use_length_bias = bool(use_length_bias)

        self.boundary_loss_weight = float(boundary_loss_weight)
        self.focal_loss_weight = float(focal_loss_weight)
        self.focal_gamma = float(focal_gamma)
        self.use_small_class_refine = bool(use_small_class_refine)
        self.refine_reg_weight = float(refine_reg_weight)
        self.auto_load_model2 = bool(auto_load_model2)
        self.model2_ckpt_path = model2_ckpt_path
        self.auto_load_model1_reference = bool(auto_load_model1_reference)
        self.model1_ckpt_path = model1_ckpt_path
        self.use_reference_ensemble = bool(use_reference_ensemble)
        self.dynamic_blend = float(dynamic_blend)
        self.dynamic_margin = float(dynamic_margin)
        self.dynamic_temperature = float(dynamic_temperature)
        self.use_final_eval_calibration = bool(use_final_eval_calibration)
        self.final_eval_calibration_scale = float(final_eval_calibration_scale)
        self.use_boundary_gated_calibration = bool(use_boundary_gated_calibration)
        self.calibration_gate_floor = float(calibration_gate_floor)
        self.calibration_gate_strength = float(calibration_gate_strength)
        self.freeze_base_model = bool(freeze_base_model)
        self.is_reference_model = bool(is_reference_model)
        self.reference_model = None

        self.dropout = nn.Dropout(dropout_rate)

        if self.use_local_context:
            self.local_context = MultiScaleLocalContext(hidden_size, dropout_rate=dropout_rate)
        else:
            self.local_context = nn.Identity()

        # GlobalPointer 主打分头
        self.dense = nn.Linear(hidden_size, self.num_labels * self.inner_dim * 2)

        # 边界预测头：[B, L, C, 2]
        self.boundary_classifier = nn.Linear(hidden_size, self.num_labels * 2)

        # 类别相关 Biaffine/双线性打分头
        self.biaffine_start = nn.Linear(hidden_size, self.num_labels * self.biaffine_dim)
        self.biaffine_end = nn.Linear(hidden_size, self.num_labels * self.biaffine_dim)

        # 可学习长度偏置：每个类别学习不同实体长度倾向
        self.length_bias = nn.Parameter(torch.zeros(self.num_labels, self.max_len_for_bias))

        # 可学习融合系数，使用 tanh 限幅，避免破坏 GlobalPointer 原始 logits
        self.boundary_scale = nn.Parameter(torch.tensor(float(boundary_logit_weight)))
        self.biaffine_scale = nn.Parameter(torch.tensor(float(biaffine_logit_weight)))
        self.length_scale = nn.Parameter(torch.tensor(0.02))

        # 第二创新点：小类友好的 Span Refinement Bias。
        # 该分支从 0 初始化，训练初期几乎等价于最终版 model1；
        # 只对边界敏感/小样本类别开放，保护“症状、处方用药”等大类优势。
        self.refine_class_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.refine_length_bias = nn.Parameter(torch.zeros(self.num_labels, self.max_len_for_bias))
        self.refine_scale = nn.Parameter(torch.tensor(0.03))

        refine_mask = torch.zeros(self.num_labels, dtype=torch.float)
        if self.num_labels >= 11:
            # 0 患者信息, 1 时间, 2 西医病名, 3 中医病名, 4 症状, 5 检查,
            # 6 检查结果, 7 治法, 8 证型, 9 处方用药, 10 方剂。
            # v9：只开放“高 Precision 但 Recall 仍有空间”的类别做极弱残差适配。
            # 不开放“检查”(5)，因为该类 Precision 容易被拉低；
            # 开放处方用药(9)，因为它样本量大且原始 Precision 高，少量召回提升对总体 F1 最划算。
            for idx in [1, 2, 3, 6, 9, 10]:
                if idx < self.num_labels:
                    refine_mask[idx] = 1.0
        else:
            refine_mask[:] = 1.0
        self.register_buffer("refine_class_mask", refine_mask, persistent=True)

        # 初始化新增模块，尽量让训练初期接近 GlobalPointer
        nn.init.xavier_uniform_(self.boundary_classifier.weight)
        nn.init.zeros_(self.boundary_classifier.bias)
        nn.init.xavier_uniform_(self.biaffine_start.weight)
        nn.init.zeros_(self.biaffine_start.bias)
        nn.init.xavier_uniform_(self.biaffine_end.weight)
        nn.init.zeros_(self.biaffine_end.bias)

        # model3 必须基于已经锁定的 model2；若找不到 model2 checkpoint，则回退到 model1。
        loaded_from_model2 = False
        if self.auto_load_model2:
            loaded_from_model2 = self._try_load_model2_checkpoint(self.model2_ckpt_path)
        if (not loaded_from_model2) and self.auto_load_model1_reference:
            self._try_load_model1_checkpoint(self.model1_ckpt_path)

        # v18：验证/测试阶段使用冻结的 model1 作为稳定参考。
        # 只让当前 model2 接管 v13 中稳定受益的类别：8=证型, 9=处方用药。
        route_mask = torch.zeros(self.num_labels, dtype=torch.float)
        if self.num_labels >= 11:
            for idx in [8, 9]:
                route_mask[idx] = 1.0
        else:
            route_mask[:] = 1.0
        self.register_buffer("ensemble_current_class_mask", route_mask, persistent=True)

        # v2 不再使用强动态路由，避免上一版 Precision 明显下降。
        # 保留该 mask 仅作兼容，默认全 0。
        dynamic_mask = torch.zeros(self.num_labels, dtype=torch.float)
        self.register_buffer("dynamic_route_class_mask", dynamic_mask, persistent=True)

        # v7：最终验证/测试阶段的长度门控召回校准。
        # 只给“当前 Precision 较高但 Recall 仍有空间”的类别加极弱正向偏置；
        # 并用长度范围限制，避免把明显不合理的长跨度误报放出来。
        final_bias = torch.zeros(self.num_labels, dtype=torch.float)
        final_min_len = torch.ones(self.num_labels, dtype=torch.long)
        final_max_len = torch.full((self.num_labels,), self.max_len_for_bias, dtype=torch.long)
        if self.num_labels >= 11:
            # 0 患者信息, 1 时间, 2 西医病名, 3 中医病名, 4 症状, 5 检查,
            # 6 检查结果, 7 治法, 8 证型, 9 处方用药, 10 方剂。
            # v11 调参重点：
            # 1) 回退 v10 对“症状、检查”的放宽，避免新增误报拉低 Precision；
            # 2) 保留 v9 的安全校准范围，只单独增强“处方用药、方剂”；
            # 3) 最终校准会经过边界置信门控，只有边界头也支持的 span 才获得更完整补偿；
            # 4) 检查类继续不放宽，因为该类边界杂、Precision 容易被拉低。
            final_bias[1] = 0.22   # 时间：沿用 v9，轻补召回
            final_bias[2] = 0.24   # 西医病名：沿用 v9，小类召回补偿
            final_bias[3] = 0.14   # 中医病名：沿用 v9，小幅补召回
            final_bias[5] = 0.00   # 检查：不放宽，保护 Precision
            final_bias[6] = 0.19   # 检查结果：沿用 v9，适度补召回
            final_bias[9] = 0.46   # 处方用药：在 v11 基础上极小幅补召回，避免 v13 过度放宽
            final_bias[10] = 0.16  # 方剂：收紧补偿，避免小类误报拉低 Precision

            # 最小长度限制用于过滤明显不合理的 1 字误报；
            # 处方用药保留 1 字，因为部分药名/简称可能较短。
            final_min_len[1] = 2
            final_min_len[2] = 2
            final_min_len[3] = 2
            final_min_len[6] = 2
            final_min_len[10] = 2

            final_max_len[1] = 14
            final_max_len[2] = 22
            final_max_len[3] = 18
            final_max_len[5] = 20
            final_max_len[6] = 45
            final_max_len[9] = 24
            final_max_len[10] = 20
        self.register_buffer("final_eval_bias", final_bias, persistent=True)
        self.register_buffer("final_eval_min_len", final_min_len, persistent=True)
        self.register_buffer("final_eval_max_len", final_max_len, persistent=True)

        if self.use_reference_ensemble and (not self.is_reference_model):
            self.reference_model = Model3GlobalPointer(
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                num_labels=num_labels,
                inner_dim=inner_dim,
                biaffine_dim=biaffine_dim,
                max_len_for_bias=max_len_for_bias,
                dropout_rate=dropout_rate,
                use_rope=use_rope,
                use_local_context=use_local_context,
                use_boundary=use_boundary,
                use_biaffine=use_biaffine,
                use_length_bias=use_length_bias,
                boundary_loss_weight=boundary_loss_weight,
                boundary_logit_weight=boundary_logit_weight,
                biaffine_logit_weight=biaffine_logit_weight,
                focal_loss_weight=focal_loss_weight,
                focal_gamma=focal_gamma,
                use_small_class_refine=False,
                refine_reg_weight=0.0,
                auto_load_model2=False,
                model2_ckpt_path=None,
                auto_load_model1_reference=auto_load_model1_reference,
                model1_ckpt_path=model1_ckpt_path,
                use_reference_ensemble=False,
                dynamic_blend=dynamic_blend,
                dynamic_margin=dynamic_margin,
                dynamic_temperature=dynamic_temperature,
                use_final_eval_calibration=False,
                final_eval_calibration_scale=final_eval_calibration_scale,
                use_boundary_gated_calibration=False,
                calibration_gate_floor=calibration_gate_floor,
                calibration_gate_strength=calibration_gate_strength,
                freeze_base_model=False,
                is_reference_model=True,
            )
            for param in self.reference_model.parameters():
                param.requires_grad = False
            self.reference_model.eval()
            print("[model3] v14 conservative：回滚 v11 稳定路线；静态接管类别: 证型、处方用药；最终校准仅做保守边界门控补偿。")

        # 保守微调：加载 model2 后冻结主体，仅训练极少数残差适配参数。
        if self.freeze_base_model and (not self.is_reference_model):
            self._freeze_base_model_keep_delta()

    def _freeze_base_model_keep_delta(self):
        """冻结 model2 主体，保留小类残差适配器可训练，防止破坏 v18 的稳定结果。"""
        trainable_keywords = ["refine_class_bias", "refine_length_bias", "refine_scale"]
        for name, param in self.named_parameters():
            keep_trainable = any(k in name for k in trainable_keywords)
            # reference_model 本来就是冻结的，这里也不解冻。
            param.requires_grad = keep_trainable
        print("[model3] 已冻结 model2 主体，仅训练小类 Span 残差适配参数。")

    # -----------------------------------------------------
    # progressive initialization from finalized model1
    # -----------------------------------------------------

    def _candidate_model2_ckpts(self, explicit_path: Optional[str] = None):
        paths = []
        if explicit_path:
            paths.append(Path(explicit_path))
        cwd = Path(os.getcwd())
        paths.append(cwd / "checkpoints" / "model2" / "best_model.pt")
        paths.append(cwd / "outputs" / "model2" / "best_model.pt")
        try:
            root = Path(__file__).resolve().parents[2]
            paths.append(root / "checkpoints" / "model2" / "best_model.pt")
        except Exception:
            pass
        seen = set()
        out = []
        for p in paths:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                out.append(p)
        return out

    def _try_load_model2_checkpoint(self, explicit_path: Optional[str] = None) -> bool:
        """优先读取最终版 model2 权重，保证 model3 建立在前两个创新点之上。"""
        ckpt_path = None
        for p in self._candidate_model2_ckpts(explicit_path):
            if p.exists():
                ckpt_path = p
                break
        if ckpt_path is None:
            print("[model3] 未找到 checkpoints/model2/best_model.pt，将尝试回退到 model1。")
            return False
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(state, dict):
                print(f"[model3] model2 checkpoint 格式无法识别，跳过加载: {ckpt_path}")
                return False
            clean_state = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                clean_state[nk] = v
            missing, unexpected = self.load_state_dict(clean_state, strict=False)
            # 注意：这里保留 checkpoint 中的 refine_class_mask，不强行覆盖。
            # 你当前最好结果来自 v11 的这一路线；v13 强行重设/放开后导致 Pred 暴涨、Precision 明显下滑。
            print(f"[model3] 已基于 model2 checkpoint 初始化: {ckpt_path}")
            print(f"[model3] 未加载参数数量: {len(missing)}；额外参数数量: {len(unexpected)}")
            return True
        except Exception as e:
            print(f"[model3] 加载 model2 checkpoint 失败，将尝试回退到 model1。原因: {e}")
            return False

    def _candidate_model1_ckpts(self, explicit_path: Optional[str] = None):
        paths = []
        if explicit_path:
            paths.append(Path(explicit_path))
        cwd = Path(os.getcwd())
        paths.append(cwd / "checkpoints" / "model1" / "best_model.pt")
        paths.append(cwd / "outputs" / "model1" / "best_model.pt")
        # also try project root inferred from this file: model/proposed/model2.py -> root
        try:
            root = Path(__file__).resolve().parents[2]
            paths.append(root / "checkpoints" / "model1" / "best_model.pt")
        except Exception:
            pass
        # de-duplicate while preserving order
        seen = set()
        out = []
        for p in paths:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                out.append(p)
        return out

    def _try_load_model1_checkpoint(self, explicit_path: Optional[str] = None):
        """
        第二创新点要求基于已确定的 model1。这里优先读取 checkpoints/model1/best_model.pt
        作为初始化；新增的 refine 参数保持随机/零初始化。若找不到 checkpoint，则自动退回从头训练。
        """
        ckpt_path = None
        for p in self._candidate_model1_ckpts(explicit_path):
            if p.exists():
                ckpt_path = p
                break
        if ckpt_path is None:
            print("[model3] 未找到 checkpoints/model1/best_model.pt，将从头训练 model3。")
            return False

        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if not isinstance(state, dict):
                print(f"[model3] model1 checkpoint 格式无法识别，跳过加载: {ckpt_path}")
                return False
            # 兼容 DataParallel 的 module. 前缀
            clean_state = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                clean_state[nk] = v
            missing, unexpected = self.load_state_dict(clean_state, strict=False)
            print(f"[model3] 已基于 model1 checkpoint 初始化: {ckpt_path}")
            print(f"[model3] 未加载参数数量: {len(missing)}；额外参数数量: {len(unexpected)}")
            return True
        except Exception as e:
            print(f"[model3] 加载 model1 checkpoint 失败，改为从头训练。原因: {e}")
            return False

    # -----------------------------------------------------
    # RoPE
    # -----------------------------------------------------
    def sinusoidal_position_embedding(self, seq_len: int, output_dim: int, device) -> torch.Tensor:
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(seq_len, output_dim)
        return embeddings[None, :, None, :]

    def apply_rope(self, tensor: torch.Tensor) -> torch.Tensor:
        _, seq_len, _, inner_dim = tensor.size()
        pos_emb = self.sinusoidal_position_embedding(seq_len, inner_dim, tensor.device)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        tensor2 = torch.stack([-tensor[..., 1::2], tensor[..., ::2]], dim=-1).reshape_as(tensor)
        return tensor * cos_pos + tensor2 * sin_pos

    # -----------------------------------------------------
    # mask / labels
    # -----------------------------------------------------
    def build_valid_token_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size = attention_mask.size(0)
        device = attention_mask.device
        valid_mask = attention_mask.clone()
        valid_mask[:, 0] = 0
        seq_lens = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=device)
        valid_mask[batch_indices, seq_lens] = 0
        return valid_mask

    def build_valid_span_mask(self, valid_token_mask: torch.Tensor) -> torch.Tensor:
        seq_len = valid_token_mask.size(1)
        span_mask = valid_token_mask[:, None, :, None] * valid_token_mask[:, None, None, :]
        tril_mask = torch.tril(torch.ones(seq_len, seq_len, device=valid_token_mask.device), diagonal=-1).bool()
        valid_span_mask = span_mask.bool() & (~tril_mask[None, None, :, :])
        return valid_span_mask

    def add_mask_tril(self, logits: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
        valid_span_mask = self.build_valid_span_mask(valid_token_mask)
        return logits.masked_fill(~valid_span_mask, -1e12)

    def normalize_labels(self, labels: torch.Tensor, seq_len: int) -> torch.Tensor:
        if labels.dim() != 4:
            raise ValueError(f"span labels 必须是 4 维，当前 shape={tuple(labels.shape)}")
        if labels.size(1) == self.num_labels:
            return labels.float()
        if labels.size(-1) == self.num_labels:
            return labels.permute(0, 3, 1, 2).contiguous().float()
        raise ValueError(
            f"labels 形状与 num_labels 不匹配，labels={tuple(labels.shape)}, "
            f"num_labels={self.num_labels}, seq_len={seq_len}"
        )

    def build_boundary_targets(self, labels: torch.Tensor):
        start_targets = labels.max(dim=-1).values.permute(0, 2, 1).contiguous()
        end_targets = labels.max(dim=-2).values.permute(0, 2, 1).contiguous()
        return start_targets, end_targets

    # -----------------------------------------------------
    # auxiliary losses
    # -----------------------------------------------------
    def boundary_loss(self, boundary_logits: torch.Tensor, labels: torch.Tensor, valid_token_mask: torch.Tensor) -> torch.Tensor:
        start_targets, end_targets = self.build_boundary_targets(labels)
        targets = torch.stack([start_targets, end_targets], dim=-1)

        # 轻量正样本加权，缓解边界正负极度不均衡。
        pos_count = targets.sum().detach()
        total_count = (valid_token_mask.sum() * self.num_labels * 2).float().clamp_min(1.0)
        neg_count = (total_count - pos_count).clamp_min(1.0)
        pos_weight_value = torch.clamp(neg_count / pos_count.clamp_min(1.0), min=1.0, max=8.0)

        bce = F.binary_cross_entropy_with_logits(boundary_logits, targets.float(), reduction="none")
        weight = torch.where(targets > 0.5, pos_weight_value, torch.ones_like(targets))
        bce = bce * weight

        mask = valid_token_mask[:, :, None, None].float()
        bce = bce * mask
        denom = mask.sum().clamp_min(1.0) * self.num_labels * 2
        return bce.sum() / denom

    def focal_span_loss(self, logits: torch.Tensor, labels: torch.Tensor, valid_span_mask: torch.Tensor) -> torch.Tensor:
        if self.focal_loss_weight <= 0:
            return logits.new_tensor(0.0)

        x = torch.clamp(logits, min=-20.0, max=20.0)
        y = labels.float()
        prob = torch.sigmoid(x)

        bce = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        pt = prob * y + (1.0 - prob) * (1.0 - y)
        focal_weight = torch.pow((1.0 - pt).clamp(min=0.0), self.focal_gamma)
        loss = bce * focal_weight

        mask = valid_span_mask.expand(-1, self.num_labels, -1, -1).float()
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)

    # -----------------------------------------------------
    # logits enhancement
    # -----------------------------------------------------
    def compute_biaffine_logits(self, sequence_output: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = sequence_output.size()
        s = self.biaffine_start(sequence_output)
        e = self.biaffine_end(sequence_output)
        s = s.view(batch_size, seq_len, self.num_labels, self.biaffine_dim)
        e = e.view(batch_size, seq_len, self.num_labels, self.biaffine_dim)
        biaffine_logits = torch.einsum("bmcd,bncd->bcmn", s, e)
        biaffine_logits = biaffine_logits / math.sqrt(self.biaffine_dim)
        return biaffine_logits

    def compute_length_bias(self, seq_len: int, device) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        start = idx.view(-1, 1)
        end = idx.view(1, -1)
        length = (end - start).clamp(min=0, max=self.max_len_for_bias - 1)
        # [C, L, L]
        bias = self.length_bias[:, length]
        return bias.unsqueeze(0)

    def compute_refine_logits(self, seq_len: int, device, dtype) -> torch.Tensor:
        idx = torch.arange(seq_len, device=device)
        start = idx.view(-1, 1)
        end = idx.view(1, -1)
        length = (end - start).clamp(min=0, max=self.max_len_for_bias - 1)
        # [C, L, L]
        len_bias = self.refine_length_bias[:, length]
        cls_bias = self.refine_class_bias.view(self.num_labels, 1, 1)
        mask = self.refine_class_mask.view(self.num_labels, 1, 1).to(device=device, dtype=dtype)
        refine = (cls_bias + len_bias) * mask
        return refine.unsqueeze(0).to(dtype=dtype)

    def apply_final_eval_calibration(self, logits: torch.Tensor, boundary_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """验证/测试阶段的长度门控召回校准。训练阶段不启用，避免改变训练目标。

        v11 新增边界置信门控：
        - 原 v8/v9 只是按类别和长度给固定正偏置；
        - v11 会参考 boundary_classifier 的 start/end 置信度；
        - 边界头也支持的 span 获得完整补偿，边界头不支持的 span 只获得较小保底补偿；
        - 目标是在增强“处方用药/方剂”召回时尽量少引入无边界依据的误报。
        """
        if self.training or (not self.use_final_eval_calibration):
            return logits
        if self.final_eval_bias.numel() != self.num_labels:
            return logits

        batch_size, num_labels, seq_len, _ = logits.size()
        device = logits.device
        dtype = logits.dtype

        idx = torch.arange(seq_len, device=device)
        start = idx.view(-1, 1)
        end = idx.view(1, -1)
        span_len = (end - start + 1).clamp(min=1, max=self.max_len_for_bias)

        min_len = self.final_eval_min_len.view(self.num_labels, 1, 1).to(device=device)
        max_len = self.final_eval_max_len.view(self.num_labels, 1, 1).to(device=device)
        length_gate = (
            (span_len.view(1, seq_len, seq_len) >= min_len)
            & (span_len.view(1, seq_len, seq_len) <= max_len)
        ).to(dtype=dtype)

        bias = self.final_eval_bias.view(self.num_labels, 1, 1).to(device=device, dtype=dtype)
        bias = self.final_eval_calibration_scale * bias * length_gate  # [C, L, L]

        if self.use_boundary_gated_calibration and boundary_logits is not None:
            # boundary_logits: [B, L, C, 2]
            start_prob = torch.sigmoid(boundary_logits[..., 0]).permute(0, 2, 1).contiguous()  # [B, C, L]
            end_prob = torch.sigmoid(boundary_logits[..., 1]).permute(0, 2, 1).contiguous()    # [B, C, L]
            boundary_gate = torch.sqrt(
                (start_prob.unsqueeze(-1) * end_prob.unsqueeze(-2)).clamp(min=1e-6)
            )  # [B, C, L, L]

            floor = max(0.0, min(1.0, self.calibration_gate_floor))
            strength = max(0.0, min(1.0, self.calibration_gate_strength))
            gate = floor + strength * boundary_gate
            gate = gate.clamp(min=0.0, max=1.0)
            return logits + bias.unsqueeze(0) * gate

        return logits + bias.unsqueeze(0)

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        if self.use_local_context:
            sequence_output = self.local_context(sequence_output, attention_mask)

        batch_size, seq_len, _ = sequence_output.size()
        valid_token_mask = self.build_valid_token_mask(attention_mask)
        valid_span_mask = self.build_valid_span_mask(valid_token_mask)

        # 1) GlobalPointer logits
        gp_output = self.dense(sequence_output)
        gp_output = gp_output.view(batch_size, seq_len, self.num_labels, self.inner_dim * 2)
        qw, kw = gp_output[..., :self.inner_dim], gp_output[..., self.inner_dim:]

        if self.use_rope:
            qw = self.apply_rope(qw)
            kw = self.apply_rope(kw)

        logits = torch.einsum("bmhd,bnhd->bhmn", qw, kw)
        logits = logits / math.sqrt(self.inner_dim)

        # 2) 边界感知偏置，使用 tanh 限幅，避免上一版 raw boundary bias 过强。
        boundary_logits = self.boundary_classifier(sequence_output)
        boundary_logits = boundary_logits.view(batch_size, seq_len, self.num_labels, 2)

        if self.use_boundary:
            start_logits = boundary_logits[..., 0].permute(0, 2, 1).contiguous()
            end_logits = boundary_logits[..., 1].permute(0, 2, 1).contiguous()
            boundary_bias = 0.5 * (torch.tanh(start_logits / 2.0).unsqueeze(-1) + torch.tanh(end_logits / 2.0).unsqueeze(-2))
            logits = logits + torch.tanh(self.boundary_scale) * boundary_bias

        # 3) Biaffine span residual logits
        if self.use_biaffine:
            biaffine_logits = self.compute_biaffine_logits(sequence_output)
            logits = logits + torch.tanh(self.biaffine_scale) * biaffine_logits

        # 4) length bias
        if self.use_length_bias:
            length_bias = self.compute_length_bias(seq_len, input_ids.device)
            logits = logits + torch.tanh(self.length_scale) * length_bias

        # 5) 小类 Span 残差适配。
        # 若启用参考融合，则先保持原始 logits，稍后加到融合后的 base_logits 上；
        # 若没有参考模型，则直接作用于当前 logits。
        refine_logits = None
        if self.use_small_class_refine:
            refine_logits = self.compute_refine_logits(seq_len, input_ids.device, logits.dtype)
            if self.reference_model is None:
                logits = logits + torch.tanh(self.refine_scale) * refine_logits

        logits = logits.masked_fill(~valid_span_mask, -1e12)

        # v18 基础：参考路由融合。
        # 大部分类别使用冻结的 model1 输出；证型/处方用药使用 model2 输出。
        # v2 第三创新点：在融合结果上只对检查/检查结果/方剂加入极弱可学习残差适配。
        if self.reference_model is not None:
            self.reference_model.eval()
            with torch.no_grad():
                ref_out = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=None,
                )
                ref_logits = ref_out["logits"].to(dtype=logits.dtype, device=logits.device)
            current_logits = logits
            route = self.ensemble_current_class_mask.view(1, self.num_labels, 1, 1).to(dtype=logits.dtype, device=logits.device)

            # 先复现 model2_v18 的稳定路由：大多数类别使用 model1，证型/处方用药使用当前模型。
            base_logits = ref_logits * (1.0 - route) + current_logits * route

            # 第三创新点：保守小类残差适配。
            # 不再做强动态路由，只在检查/检查结果/方剂上学习一个很小的 span 长度/类别残差。
            if refine_logits is not None:
                logits = base_logits + torch.tanh(self.refine_scale) * refine_logits
            else:
                logits = base_logits
            logits = logits.masked_fill(~valid_span_mask, -1e12)

        # v7：最终只在验证/测试阶段启用的召回校准。
        logits = self.apply_final_eval_calibration(logits, boundary_logits=boundary_logits)
        logits = logits.masked_fill(~valid_span_mask, -1e12)

        if labels is not None:
            labels = self.normalize_labels(labels, seq_len)
            labels = labels * valid_span_mask.float()

            gp_loss = multilabel_categorical_crossentropy(logits, labels)

            b_loss = self.boundary_loss(boundary_logits, labels, valid_token_mask) if self.use_boundary else logits.new_tensor(0.0)
            f_loss = self.focal_span_loss(logits, labels, valid_span_mask)

            refine_reg = logits.new_tensor(0.0)
            if self.use_small_class_refine and self.refine_reg_weight > 0:
                refine_reg = (self.refine_class_bias.pow(2).mean() + self.refine_length_bias.pow(2).mean())

            loss = gp_loss + self.boundary_loss_weight * b_loss + self.focal_loss_weight * f_loss + self.refine_reg_weight * refine_reg

            return {
                "loss": loss,
                "logits": logits,
                "gp_loss": gp_loss.detach(),
                "boundary_loss": b_loss.detach(),
                "focal_loss": f_loss.detach(),
                "refine_reg": refine_reg.detach(),
            }

        return {"logits": logits}


GlobalPointer = Model3GlobalPointer
Model = Model3GlobalPointer
SpanLabelingModel = Model3GlobalPointer


def build_model(
    pretrained_model_name_or_path: str,
    num_labels: int,
    dropout_rate: float = 0.1,
    inner_dim: int = 64,
    biaffine_dim: int = 32,
    boundary_loss_weight: float = 0.04,
    boundary_logit_weight: float = 0.07,
    biaffine_logit_weight: float = 0.10,
    focal_loss_weight: float = 0.010,
    use_small_class_refine: bool = True,
    refine_reg_weight: float = 6e-5,
    auto_load_model2: bool = True,
    model2_ckpt_path: Optional[str] = None,
    auto_load_model1_reference: bool = True,
    model1_ckpt_path: Optional[str] = None,
    use_reference_ensemble: bool = True,
    dynamic_blend: float = 0.0,
    dynamic_margin: float = 0.35,
    dynamic_temperature: float = 5.0,
    use_final_eval_calibration: bool = True,
    final_eval_calibration_scale: float = 1.0,
    use_boundary_gated_calibration: bool = True,
    calibration_gate_floor: float = 0.55,
    calibration_gate_strength: float = 0.45,
    freeze_base_model: bool = True,
    **kwargs,
):
    return Model3GlobalPointer(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        num_labels=num_labels,
        inner_dim=inner_dim,
        biaffine_dim=biaffine_dim,
        dropout_rate=dropout_rate,
        boundary_loss_weight=boundary_loss_weight,
        boundary_logit_weight=boundary_logit_weight,
        biaffine_logit_weight=biaffine_logit_weight,
        focal_loss_weight=focal_loss_weight,
        use_small_class_refine=use_small_class_refine,
        refine_reg_weight=refine_reg_weight,
        auto_load_model2=auto_load_model2,
        model2_ckpt_path=model2_ckpt_path,
        auto_load_model1_reference=auto_load_model1_reference,
        model1_ckpt_path=model1_ckpt_path,
        use_reference_ensemble=use_reference_ensemble,
        dynamic_blend=dynamic_blend,
        dynamic_margin=dynamic_margin,
        dynamic_temperature=dynamic_temperature,
        use_final_eval_calibration=use_final_eval_calibration,
        final_eval_calibration_scale=final_eval_calibration_scale,
        use_boundary_gated_calibration=use_boundary_gated_calibration,
        calibration_gate_floor=calibration_gate_floor,
        calibration_gate_strength=calibration_gate_strength,
        freeze_base_model=freeze_base_model,
        **kwargs,
    )
