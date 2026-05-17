# LC-TCMNER

面向肺癌中医医案的命名实体识别数据集与 BCA-GlobalPointer 模型

---

## 项目简介

本项目公开了论文《面向肺癌中医医案的命名实体识别数据集构建与BCA-GlobalPointer模型研究》中使用的：

- LC-TCMNER 数据集
- BCA-GlobalPointer 模型代码
- 训练脚本

本研究面向肺癌中医医案命名实体识别任务，针对复杂实体边界与类别分布不均衡问题，提出融合边界感知与类别自适应机制的 BCA-GlobalPointer 模型。

---

## 数据集简介

### LC-TCMNER 数据集

LC-TCMNER 是面向肺癌中医医案场景构建的中文医学命名实体识别数据集。

数据集包含：

- 942 个肺癌中医医案
- 4249 条切分样本
- 11 类实体
- 69899 个实体标注

---

## 实体类别

| 实体类型 | 含义 |
|---|---|
| PAT | 患者信息 |
| TIME | 时间 |
| WEST_DIS | 西医病名 |
| TCM_DIS | 中医病名 |
| SYM | 症状 |
| EXAM | 检查 |
| EXAM_RES | 检查结果 |
| TREAT | 治法 |
| SYNDROME | 证型 |
| PRESCRIPTION | 处方用药 |
| FORMULA | 方剂 |

## 环境配置

推荐环境：

```bash
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2

transformers==4.36.2
tokenizers==0.15.0

numpy==1.24.4
scikit-learn==1.3.2
tqdm==4.66.1

sentencepiece==0.1.99
accelerate==0.25.0
```

```
checkpoints内容过大无法上传，建议运行方式从model1开始，最后运行BCA-GlobalPointer
```

## 声明

本项目仅供学术研究使用。

数据集来源于公开医学文献整理与人工标注，请勿用于商业用途。
