import argparse
import requests
import time
from tqdm import tqdm
import torch
import gc
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
import os

def get_tensor_parallel_size(model_name):
    """Determines how many GPUs are needed per model based on VRAM."""
    model_name_lower = model_name.lower()
    
    if not torch.cuda.is_available():
        return 4 
        
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if model_name_lower == "qwen3.5-397b-a17b-fp8":
        if vram_gb >= 130:
            print(f"[INFO] Detected {vram_gb:.1f}GB VRAM per GPU. Spreading 397B model across 4 GPUs.")
            return 4
        elif vram_gb >= 80:
            print(f"[INFO] Detected {vram_gb:.1f}GB VRAM per GPU. Spreading 397B model across 8 GPUs.")
            return 8
        else:
            print(f"[WARNING] Detected {vram_gb:.1f}GB VRAM per GPU. Defaulting to 16 GPUs for 397B model.")
            return 16
            
    elif model_name_lower == "qwen3.5-122b-a10b-fp8":
        if vram_gb >= 130:
            print(f"[INFO] Detected {vram_gb:.1f}GB VRAM per GPU. Spreading 122B model across 2 GPUs.")
            return 2
        elif vram_gb >= 80:
            print(f"[INFO] Detected {vram_gb:.1f}GB VRAM per GPU. Spreading 122B model across 2 GPUs.")
            return 2
        elif vram_gb >= 48:
            print(f"[INFO] Detected {vram_gb:.1f}GB VRAM per GPU. Spreading 122B model across 4 GPUs.")
            return 4
        else:
            print(f"[WARNING] Detected {vram_gb:.1f}GB VRAM per GPU. Defaulting to 4 GPUs.")
            return 4
            
    return 1 

def get_llm_and_tokenizer(model_name, gpu_memory_utilization, tensor_parallel_size=1, max_model_len=8192):
    model_name_hf = ""
    model_name_lower = model_name.lower()
    
    if model_name_lower == "gemma-3-27b":
        model_name_hf  = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
        tokenizer = AutoProcessor.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "llama-3.3-70b":
        model_name_hf = "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "qwen3-30b":
        model_name_hf = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "qwen3-32b":
        model_name_hf = "RedHatAI/Qwen3-32B-FP8-dynamic"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "deepseek-r1-70b":
        model_name_hf = "RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "gpt-oss-20b":
        model_name_hf = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "glm-z1-32b":
        model_name_hf = "duydq12/GLM-Z1-32B-0414-FP8-dynamic"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "qwen3.5-35b-a3b-fp8":
        model_name_hf = "Qwen/Qwen3.5-35B-A3B-FP8"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "qwen3.6-35b-a3b-fp8":
        model_name_hf = "Qwen/Qwen3.6-35B-A3B-FP8"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "qwen3.5-122b-a10b-fp8":
        model_name_hf = "Qwen/Qwen3.5-122B-A10B-FP8"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name_lower == "qwen3.5-397b-a17b-fp8":
        model_name_hf = "Qwen/Qwen3.5-397B-A17B-FP8"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    else:
        raise Exception("Please chose one of the models: gemma-3-27b, llama-3.3-70b, qwen3-30b, qwen3-32b, deepseek-r1-70b, gpt-oss-20b, glm-z1-32b, qwen3.5-122b-a10b-fp8, qwen3.5-397b-a17b-fp8")

    current_max_len = max_model_len
    min_len = 1024
    llm = None
    
    while current_max_len >= min_len:
        llm_kwargs = {
            "model": model_name_hf,
            "trust_remote_code": True,
            "max_model_len": int(current_max_len),
            "tensor_parallel_size": tensor_parallel_size
        }

        if gpu_memory_utilization > 0.0:
            llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
            
        try:
            print(f"[INFO] Attempting to load LLM with max_model_len = {int(current_max_len)}...")
            llm = LLM(**llm_kwargs)
            print(f"[SUCCESS] LLM successfully loaded with max_model_len = {int(current_max_len)}.")
            break
        except Exception as e:
            print(f"[WARNING] Failed to load LLM with max_model_len = {int(current_max_len)}. Error: {e}")
            print("[INFO] Reducing max_model_len by 25% and retrying...")
            current_max_len = int(current_max_len * 0.75)
            
    if llm is None:
        raise RuntimeError(f"Could not load the model even with minimum context length of {min_len}.")
    
    return llm, tokenizer, int(current_max_len)

def get_pompt_chat(tokenizer, prompt_text, model_name=""):
    chat = []
    
    if "deepseek" in model_name.lower():
        chat.append({
            "role": "system",
            "content": "You are a helpful AI assistant. You must first think step-by-step about the problem. Put your entire thinking process completely inside <think> and </think> tags. Only after the </think> tag, provide your final answer."
        })
        
    if "gemma" in str(type(tokenizer)).lower() or "gemma" in model_name.lower():
        chat.extend([
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": []}
        ])
    else:
        chat.append(
            {"role": "user", "content": prompt_text}
        )
    return chat

def get_prompt(tokenizer, text, template, max_model_len, model_name="", output_reservation_length=500):
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    text = text if text is not None else ""
    prompt_text = template + text
    
    chat = get_pompt_chat(tokenizer, prompt_text, model_name)

    template_kwargs = {"add_generation_prompt": True}
    if "qwen" in model_name.lower():
        template_kwargs["enable_thinking"] = True  
        
    if len(prompt_text.split(" ")) > (max_model_len/2):
        tokens = tokenizer.apply_chat_template(chat, tokenize=True, **template_kwargs)
        if type(tokens[0]) == type(0):
            len_tokens = len(tokens)
        else:
            len_tokens = len(tokens[0])
                             
        overhead = max_model_len - (len_tokens + output_reservation_length)
        if overhead < 0:
            text = tokenizer.decode(tokenizer(text, add_special_tokens=False).input_ids[:overhead], skip_special_tokens=True)
            prompt_text = template + text
            chat = get_pompt_chat(tokenizer, prompt_text, model_name)
            result = tokenizer.apply_chat_template(chat, tokenize=False, **template_kwargs)
            return result
    
    result = tokenizer.apply_chat_template(chat, tokenize=False, **template_kwargs)
    return result

def process_batch(llm, tokenizer, actual_max_len, model_name, template, batch_data):
    texts = [item["text"] for item in batch_data]
    indices = [item["index"] for item in batch_data]
    
    prompts = [get_prompt(tokenizer, text, template, actual_max_len, model_name) for text in texts]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=actual_max_len), use_tqdm=True)
    
    results = []
    for idx, x in zip(indices, outputs):
        raw_text = x.outputs[0].text
        raw_text = raw_text.replace("assistantfinal", "</think>")
        
        if "</think>" in raw_text:
            parts = raw_text.split("</think>")
            reasoning = parts[0].replace("<think>", "").strip()
            final_answer = parts[1].replace("```json", "").replace("```", "").strip()
        else:
            reasoning = ""
            final_answer = raw_text.replace("```json", "").replace("```", "").strip()
            
        results.append({
            "index": idx,
            "output": final_answer,
            "reasoning": reasoning
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master_host', type=str, default='localhost')
    parser.add_argument('--master_port', type=int, default=8192)
    parser.add_argument('--gpu_ids', type=str, default='', help='e.g., "0,1" to restrict client to specific GPUs')
    args = parser.parse_args()

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print(f"[CLIENT] Constrained to GPUs: {args.gpu_ids}")

    host = args.master_host
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    master_url = f"{host}:{args.master_port}"

    print(f"[CLIENT] Connecting to Master at {master_url}...")

    current_config = None
    llm = None
    tokenizer = None
    actual_max_len = None
    batches_processed = 0

    while True:
        # Request batch from Master
        try:
            resp = requests.get(f"{master_url}/get_batch")
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[CLIENT] Error connecting to Master: {e}. Retrying in 10s...")
            time.sleep(10)
            continue

        batch_id = data.get("batch_id")
        if not batch_id:
            print("[CLIENT] No batches available. Waiting 5s...")
            time.sleep(5)
            continue
            
        batch_data = data.get("data", [])
        new_config = data.get("config", {})

        # MEMORY MANAGEMENT: Load or Reload Model if config changes
        config_changed = (
            current_config is None or 
            current_config.get("model") != new_config.get("model") or
            current_config.get("max_model_len") != new_config.get("max_model_len") or
            current_config.get("gpu_memory_utilization") != new_config.get("gpu_memory_utilization")
        )

        if config_changed:
            print(f"\n[CLIENT] Model config update detected. Loading {new_config['model']}...")
            
            if llm is not None:
                print("[CLIENT] Freeing previous model from memory...")
                del llm
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            tensor_parallel_size = get_tensor_parallel_size(new_config['model'])
            llm, tokenizer, actual_max_len = get_llm_and_tokenizer(
                new_config['model'], 
                new_config['gpu_memory_utilization'], 
                tensor_parallel_size, 
                new_config['max_model_len']
            )
            current_config = new_config
            print("[CLIENT] Model successfully cached in memory.")

        # PROCESSING
        print(f"\n[CLIENT] Starting Task {batch_id} ({len(batch_data)} items)...")
        start_time = time.time()
        
        results = process_batch(
            llm, 
            tokenizer, 
            actual_max_len, 
            current_config['model'], 
            current_config['template'], 
            batch_data
        )

        elapsed_time = time.time() - start_time
        batches_processed += 1
        
        print(f"[CLIENT] Finished Task {batch_id} in {elapsed_time:.2f}s. "
              f"| Total tasks processed by this client: {batches_processed}")

        # SUBMISSION
        try:
            submit_resp = requests.post(f"{master_url}/submit_batch", json={
                "batch_id": batch_id,
                "results": results
            })
            submit_resp.raise_for_status()
            print(f"[CLIENT] Successfully submitted Task {batch_id}.")
        except requests.exceptions.HTTPError as e:
            # Catch 400 Bad Request specifically (which means it timed out and was rejected)
            if e.response.status_code == 400:
                print(f"[CLIENT] ⚠️ Task {batch_id} was rejected by Master (likely timed out). Discarding results.")
            else:
                print(f"[CLIENT] HTTP Error submitting to Master: {e}")
        except requests.exceptions.RequestException as e:
            print(f"[CLIENT] Connection Error submitting to Master: {e}.")

if __name__ == '__main__':
    main()