# 每任务独立 LoRA Baseline

这个目录实现的是“每任务独立 LoRA”上界对照：

```text
base Qwen + agnews  -> agnews adapter
base Qwen + amazon  -> amazon adapter
base Qwen + yelp    -> yelp adapter
base Qwen + dbpedia -> dbpedia adapter
base Qwen + yahoo   -> yahoo adapter
```

每个任务都从同一个原始 base model 开始训练，并初始化一个新的 LoRA。不同任务之间不继承、不合并、不复用 adapter。

## 对齐 O-LoRA

本工程保留我们自己的 **Qwen2.5-3B-Instruct + PEFT LoRA + 每任务独立训练框架**，但这些部分尽量对齐 O-LoRA 官方实现：

```text
数据目录: O-LoRA/CL_Benchmark
样本结构: Task / Dataset / subset / Samples / Instance
instruction: configs/instruction_config.json 里的 SC / TC 模板
decoder 输入: BOS + instruction，训练时拼 label + EOS
loss mask: prompt 部分不算 loss，只算 label 部分
推理后处理: decode 后取 Answer: 后面的文本
评测指标: normalized exact match + rouge1 + rougeL
输出文件: predict_eval_predictions.jsonl 和 predict_results.json
```

参考官方仓库：

```text
https://github.com/cmnfriend/O-LoRA
```

## 当前模型

默认使用：

```text
Qwen/Qwen2.5-3B-Instruct
```

配置文件：

```text
configs/clora_5task_qwen25_3b.json
```

AutoDL 云端可以联网时，可以直接保持这个模型名，运行时会自动下载。也可以改成本地模型路径：

```json
"base_model_name_or_path": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
```

## 目录结构

```text
per_task_lora_baseline/
  configs/
    clora_5task_qwen25_3b.json
  scripts/
    prepare_olora_benchmark.py
    train_per_task_lora.py
    eval_per_task_lora.py
    infer_one.py
  src/
    config.py
    data.py
    metrics.py
    modeling.py
    task_specs.py
    utils.py
  requirements.txt
  README.md
```

## AutoDL 环境准备

```bash
cd /root/autodl-tmp/per_task_lora_baseline
pip install -r requirements.txt
```

下载 O-LoRA 官方 `CL_Benchmark` 数据：

```bash
python scripts/prepare_olora_benchmark.py
```

下载后目录应该是：

```text
CL_Benchmark/
  SC/
    amazon/
    yelp/
  TC/
    agnews/
    dbpedia/
    yahoo/
```

## 默认任务

当前默认 5 个任务与 O-LoRA 官方目录名一致：

```text
agnews  -> AG News，新闻主题分类
amazon  -> Amazon Reviews，商品评论 5 分类情感分类
yelp    -> Yelp Reviews，餐饮评论 5 分类情感分类
dbpedia -> DBpedia，百科条目分类
yahoo   -> Yahoo Answers，问题主题分类
```

注意：这里的 `amazon` 和 `yelp` 是 O-LoRA/CL benchmark 的 5 分类版本，不是 `amazon_polarity` / `yelp_polarity` 二分类。

## 训练

训练全部 5 个独立 LoRA：

```bash
python scripts/train_per_task_lora.py --config configs/clora_5task_qwen25_3b.json
```

只训练指定任务：

```bash
python scripts/train_per_task_lora.py --config configs/clora_5task_qwen25_3b.json --tasks agnews,amazon
```

训练输出目录：

```text
outputs/per_task_lora/clora5_qwen25_3b/
  agnews/adapter/
  amazon/adapter/
  yelp/adapter/
  dbpedia/adapter/
  yahoo/adapter/
```

## 评估

评估全部任务：

```bash
python scripts/eval_per_task_lora.py \
  --config configs/clora_5task_qwen25_3b.json \
  --adapters_root outputs/per_task_lora/clora5_qwen25_3b
```

每个任务会输出：

```text
outputs/per_task_lora/clora5_qwen25_3b/{task}/predict_eval_predictions.jsonl
outputs/per_task_lora/clora5_qwen25_3b/{task}/predict_results.json
```

`predict_results.json` 里主要看：

```text
predict_exact_match
```

它就是 normalized exact match，对分类任务可理解为 accuracy。

## 单条样本推理

AG News 示例：

```bash
python scripts/infer_one.py \
  --config configs/clora_5task_qwen25_3b.json \
  --task agnews \
  --adapter_path outputs/per_task_lora/clora5_qwen25_3b/agnews/adapter \
  --sentence "Stocks rose on Monday as technology companies led a broad rally." \
  --label Business
```

Yelp 示例：

```bash
python scripts/infer_one.py \
  --config configs/clora_5task_qwen25_3b.json \
  --task yelp \
  --adapter_path outputs/per_task_lora/clora5_qwen25_3b/yelp/adapter \
  --sentence "The food was great, but the service was slow." \
  --label neutral
```

如果传了 `--label`，脚本会额外输出 normalized exact match 是否正确。

## 说明

- 本地代码不会下载模型或数据。
- 下载只会发生在 AutoDL 上运行 `prepare_olora_benchmark.py` 或训练/评估脚本时。
- `.gitignore` 已排除 `CL_Benchmark/`、模型权重、adapter 输出、PDF 和 DOCX。
