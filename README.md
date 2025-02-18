# deepseek-sglang-aws

This repository launches the `SGLang` server with the DeepSeek-R1-Distilled model (tested on Qwen-1.5B).

Large language models (LLMs) are increasingly used for complex tasks that require multiple generation calls, advanced prompting techniques, control flow, and structured inputs/outputs. However, efficient systems are lacking for programming and executing these applications. We introduce `SGLang`, a system for efficient execution of complex language model programs. 

`SGLang` consists of a frontend language and a runtime. The frontend simplifies programming with primitives for generation and parallelism control. The runtime accelerates execution with novel optimizations like `RadixAttention` for KV cache reuse and compressed finite state machines for faster structured output decoding. Experiments show that `SGLang` achieves up to 6.4x higher throughput compared to state-of-the-art inference systems.

## Steps to run

Follow the steps below to run this sample and deploy distilled deepseek models using `SGLang`.

1. Create and activate a python3.12 environment:

    ```bash
    uv venv .sglang_deepseek --python 3.12
    source .sglang_deepseek/bin/activate
    ```

1. Install all the requirements to serve the model:

    ```bash
    uv pip install -r <(uv pip compile pyproject.toml)
    ```

1. Pull the latest version of `sglang`:

    ```bash
    docker pull lmsysorg/sglang:latest
    ```

1. Set the environment variables (model id, inference parameters):

    ```bash
    export MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    export SERVER_PORT="30000"
    export TEMPERATURE="0.1"
    export MAX_TOKENS="2048"
    export TP_DEGREE="1"
    ```

1. Run the script and deploy the model:

    ```bash
    python sglang_deepseek_deployment.py
    ```




