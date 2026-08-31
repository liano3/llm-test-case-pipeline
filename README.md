# LLM Test Case Pipeline

使用大模型为编程题生成参考解、输入约束和候选输入，再通过多程序共识得到期望输出并筛选测试用例。

这里只保留通用源码，不包含数据集、模型、运行结果、提示词轨迹或特定任务信息。

## 安装

需要 Python 3.10+、一个 OpenAI 兼容接口，以及可选的本地模型服务：

```bash
pip install "openai>=1.0.0"
pip install huggingface_hub vllm
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

## 本地模型

```bash
python model/download.py
bash model/serve.sh
```

模型下载到 `model/Qwen3.5-27B/`。默认使用一张 GPU；多卡可使用：

```bash
GPU_IDS=0,1 TP_SIZE=2 bash model/serve.sh
```

## 数据格式

题目文件是 JSON 数组：

```json
[
  {
    "id": "problem-id",
    "question": "完整题面",
    "solutions": ["候选 Python 程序 1", "候选 Python 程序 2"]
  }
]
```

合成的输入和输出示例分别位于 `data/problems.example.json` 和
`data/cases.example.jsonl`。

## 使用

```bash
mkdir -p artifacts outputs

python src/generate_references.py \
  --problems data/problems.example.json \
  --output artifacts/references.jsonl \
  --model MODEL_NAME

python src/generate_constraints.py \
  --problems data/problems.example.json \
  --output artifacts/constraints.jsonl \
  --model MODEL_NAME

python src/generate_inputs.py \
  --problems data/problems.example.json \
  --references artifacts/references.jsonl \
  --constraints artifacts/constraints.jsonl \
  --output artifacts/inputs.jsonl \
  --model MODEL_NAME

python src/execute_consensus.py \
  --problems data/problems.example.json \
  --inputs artifacts/inputs.jsonl \
  --references artifacts/references.jsonl \
  --checkpoint artifacts/executions.jsonl

python src/validate_inputs.py \
  --problems data/problems.example.json \
  --executions artifacts/executions.jsonl \
  --constraints artifacts/constraints.jsonl \
  --checkpoint artifacts/validation.jsonl

python src/build_consensus.py \
  --problems data/problems.example.json \
  --executions artifacts/executions.jsonl \
  --validation artifacts/validation.jsonl \
  --output outputs/consensus.json

python src/export_cases.py \
  --problems data/problems.example.json \
  --consensus-report outputs/consensus.json \
  --output outputs/cases.jsonl
```

各脚本可通过 `--help` 查看参数。若部分题目生成失败，可使用
`src/recover_missing_inputs.py` 恢复。

注意：流水线会执行模型生成代码和候选程序，请在隔离的容器或虚拟机中运行。
