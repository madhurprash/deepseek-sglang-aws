import os
import sys
import time
import json
import logging
import requests
import subprocess

"""
This script launches the SGLang server with the DeepSeek-R1-Distilled model (tested on Qwen-1.5B).

Large language models (LLMs) are increasingly used for complex tasks that require multiple generation calls, advanced prompting techniques, control flow, 
and structured inputs/outputs. However, efficient systems are lacking for programming and executing these applications. We introduce SGLang, a system for 
efficient execution of complex language model programs. 

SGLang consists of a frontend language and a runtime. The frontend simplifies programming with primitives for generation and parallelism control. The runtime 
accelerates execution with novel optimizations like RadixAttention for KV cache reuse and compressed finite state machines for faster structured output decoding. 
Experiments show that SGLang achieves up to 6.4x higher throughput compared to state-of-the-art inference systems.
"""

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load environment variables. This includes the model id, server port, temperature, max tokens
# and the tp degree
MODEL_ID = os.getenv("MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
SERVER_PORT = os.getenv("SERVER_PORT", "30000")
TEMPERATURE = os.getenv("TEMPERATURE", "0.1")
MAX_TOKENS = os.getenv("MAX_TOKENS", "64")
TP_DEGREE = os.getenv("TP_DEGREE", "1")


def launch_server():
    container_name = "sglang_deepseek_container"
    home = os.path.expanduser("~")

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--shm-size", "32g",
        "-p", f"{SERVER_PORT}:{SERVER_PORT}",
        "-v", f"{home}/.cache/huggingface:/root/.cache/huggingface",
        "--ipc", "host",
        "--network", "host",
        "--privileged",
        "--name", container_name,
        "lmsysorg/sglang:latest",
        "python3", "-m", "sglang.launch_server",
        "--model", MODEL_ID,
        "--tp", TP_DEGREE,
        "--trust-remote-code",
        "--port", SERVER_PORT,
    ]

    logger.info("Starting Docker container with SGLang server...")
    logger.info("Command: " + " ".join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process, container_name


def wait_for_server(timeout=300):
    start_time = time.time()
    url = f"http://127.0.0.1:{SERVER_PORT}/get_model_info"
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info("Server ready!")
                return True
        except Exception as e:
            logger.debug(f"Server not ready: {e}")
        time.sleep(2)
    return False


def run_inference():
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "List 3 countries and their capitals."}
        ],
        "temperature": float(TEMPERATURE),
        "max_tokens": int(MAX_TOKENS),
    }
    curl_cmd = [
        "curl",
        f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-H", "Authorization: Bearer EMPTY",
        "-d", json.dumps(payload)
    ]

    logger.info("Sending inference request with curl...")
    result = subprocess.run(curl_cmd, capture_output=True, text=True)
    logger.info("Inference response:")
    logger.info(result.stdout)
    if result.stderr:
        logger.error(f"Inference error: {result.stderr}")


def main():
    # Launch the sg lang server
    server_proc, container_name = launch_server()
    logger.info("Waiting for the server to become ready...")
    if not wait_for_server():
        logger.error("Server did not become ready within the timeout period.")
        subprocess.run(["docker", "logs", container_name])

    logger.info("Server is up and running!")
    logger.info(f"Going to run inference against the model:")
    run_inference()


if __name__ == "__main__":
    main()