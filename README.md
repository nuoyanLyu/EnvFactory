# EnvFactory

This is the official implementation of **EnvFactory: Scaling Tool-Use Agents via Executable Environments Synthesis and Robust RL**.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Construction](#environment-construction)
- [Tool-use Trajectories Synthesis](#tool-use-trajectories-synthesis)
- [Data Processing](#data-processing)
- [SFT Training](#sft)
- [RL Training](#rl)

## Quick Start
### Environment Setup
```bash
conda create -n EnvFactory python=3.12
conda activate EnvFactory

git clone https://github.com/LARK-AI-Lab/EnvFactory
cd EnvFactory

pip install -e ".[sglang]"
```

### Set Environment Variables
Create a `.env` file with your API keys as follows:

```dotenv
# 1. Default MCP configuration path
MCP_CONFIG_PATH=configs/mcp_server.json

# 2. General embedding and chat models
EMBEDDING_URL=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_API_KEY=...
EMBEDDING_MODEL=BAAI/bge-m3

CHAT_URL=https://api.deepseek.com
CHAT_API_KEY=...
CHAT_MODEL=deepseek-chat

# 3. Provider configuration (pick one)
# DeepSeek
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 4. SGLang or vLLM
SGLANG_BASE_URL_PORT=8000
SGLANG_BASE_URL=http://localhost:${SGLANG_BASE_URL_PORT}/v1
SGLANG_API_KEY=placeholder
SGLANG_MODEL=Qwen/Qwen3-30B-A3B-Thinking-2507
```

- Refer to `PROVIDER_MAPPING` in [`src/gen/__init__.py`](src/gen/__init__.py) for supported providers.
- Use [`src/serve/sglang.sh`](src/serve/sglang.sh) and [`src/serve/vllm.sh`](src/serve/vllm.sh) to serve local models.

## Environment Construction
TODO (@Zilin)

## Tool-use Trajectories Synthesis
1. Follow the example in [`examples/load_tool_graph.ipynb`](examples/load_tool_graph.ipynb) to save `graph.pkl` locally.
2. Run [`examples/synthesize_query.py`](examples/sythesize_query.py) to synthesize tool-use trajectories.

## Data Processing
After generation, run [`examples/process_data.sh`](examples/process_data.sh) to convert the data into SFT and RL training formats:

```bash
bash examples/process_data.sh
```

To visualize the generation pipeline, run [`src/app/app.py`](src/app/app.py).

## SFT
We use [LlamaFactory](https://github.com/hiyouga/LLaMAFactory) for SFT training. Add the following entry to your dataset config:

```json
"env_factory_sft": {
    "file_name": "env_factory_sft.json",
    "formatting": "alpaca",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "history": "history",
        "system": "system"
    }
}
```

The SFT configuration is provided in [`configs/llamafactory_sft.yaml`](configs/llamafactory_sft.yaml).

## RL
We use the forked [VeRL](https://github.com/RolandXMR/verl) for RL training. Please refer to the [README](https://github.com/RolandXMR/verl/tree/main/EnvFactory) for files we modified.

## Citation
If you find our work helpful, please consider citing: