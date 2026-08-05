# 每任务独立 LoRA Baseline

这个目录实现的是“每任务独立 LoRA”对照实验：

```text
base model + 任务 1 数据 -> 任务 1 的 LoRA adapter
base model + 任务 2 数据 -> 任务 2 的 LoRA adapter
base model + 任务 3 数据 -> 任务 3 的 LoRA adapter
...
```

每个任务都从同一个原始 base model 开始训练，并初始化一个新的 LoRA。不同任务之间不继承、不合并、不复用 adapter。

## 当前模型

默认使用：

```text
Qwen/Qwen2.5-3B-Instruct
```

配置文件是：

```text
configs/clora_5task_qwen25_3b.json
```

如果 AutoDL 云端可以联网，可以直接保持这个模型名，运行时会自动下载。也可以改成本地模型路径，例如：

```json
"base_model_name_or_path": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
```

## 目录结构

```text
per_task_lora_baseline/
  configs/
    clora_5task_qwen25_3b.json
  scripts/
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

在云端机器上安装依赖：

```bash
cd /root/autodl-tmp/per_task_lora_baseline
pip install -r requirements.txt
```

## 默认数据集

当前默认使用 5 个任务：

```text
ag_news  -> AG News，新闻主题分类
amazon   -> Amazon Reviews，商品评论情感分类
yelp     -> Yelp Reviews，餐饮评论情感分类
dbpedia  -> DBpedia，百科条目分类
yahoo    -> Yahoo Answers，问题主题分类
```

对应 HuggingFace 数据集：

```text
ag_news
amazon_polarity
yelp_polarity
dbpedia_14
yahoo_answers_topics
```

## 训练

训练全部 5 个独立 LoRA：

```bash
python scripts/train_per_task_lora.py --config configs/clora_5task_qwen25_3b.json
```

只训练指定任务：

```bash
python scripts/train_per_task_lora.py --config configs/clora_5task_qwen25_3b.json --tasks ag_news,amazon
```

训练输出目录：

```text
outputs/per_task_lora/clora5_qwen25_3b/
  ag_news/adapter/
  amazon/adapter/
  yelp/adapter/
  dbpedia/adapter/
  yahoo/adapter/
```

每个 `adapter/` 都是一个独立 LoRA 文件夹。评估哪个任务，就加载哪个任务对应的 adapter。

## 评估

评估全部任务：

```bash
python scripts/eval_per_task_lora.py \
  --config configs/clora_5task_qwen25_3b.json \
  --adapters_root outputs/per_task_lora/clora5_qwen25_3b
```

评估脚本会按任务加载：

```text
原始 Qwen2.5-3B-Instruct + 当前任务 adapter
```

然后在该任务测试集上计算 accuracy。

## 单条样本推理

例如测试一条 AG News 新闻：

```bash
python scripts/infer_one.py \
  --config configs/clora_5task_qwen25_3b.json \
  --task ag_news \
  --adapter_path outputs/per_task_lora/clora5_qwen25_3b/ag_news/adapter \
  --text "Stocks rose on Monday as technology companies led a broad rally."
```

测试 Yelp 情感分类：

```bash
python scripts/infer_one.py \
  --config configs/clora_5task_qwen25_3b.json \
  --task yelp \
  --adapter_path outputs/per_task_lora/clora5_qwen25_3b/yelp/adapter \
  --text "The food was great, but the service was slow."
```

## 训练方式说明

现在代码使用 Qwen2.5-3B-Instruct 的 causal LM 训练方式，不再使用 T5 的 seq2seq 训练方式。

每条样本会被处理成：

```text
用户 prompt + 标准答案 label
```

loss 只计算答案 label 部分，prompt 部分会被 mask 掉。

LoRA 默认挂在 Qwen 常用线性层：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

## 说明

- 这个 baseline 用来作为固定预算 LoRA 的性能上界对照。
- 它的优点是每个任务有自己的 LoRA，基本不会互相覆盖。
- 它的缺点是任务越多，adapter 数量越多，参数量和部署管理成本线性增长。
- 本地代码不会下载模型或数据。下载只会发生在你把代码放到 AutoDL 并运行训练/评估脚本时。
