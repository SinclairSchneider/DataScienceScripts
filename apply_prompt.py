from datasets import load_dataset
import math
import torch
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from transformers import AutoProcessor, AutoTokenizer
import json
from tqdm import tqdm
import multiprocessing as mp
import multiprocessing.pool as mpp
import argparse
import os
import pandas as pd
import ctypes, signal

class _NoDaemonProcess(mp.Process):
    """A Process subclass that is *not* daemonic **and** is compatible with
    the way `multiprocessing.pool` constructs workers since Python 3.12.

    The pool implementation calls `Process(ctx, …)` where the **first** positional
    argument is the *context* object, *not* the traditional `group` parameter.
    We therefore strip that extra leading argument off before delegating to the
    real `multiprocessing.Process.__init__`.
    """

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], mp.context.BaseContext):
            args = args[1:]
        super().__init__(*args, **kwargs)

        # Tell kernel to send SIGKILL if parent dies
        libc = ctypes.CDLL(None)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):  
        pass

class NonDaemonPool(mpp.Pool):
    """A `multiprocessing.Pool` whose workers may spawn child processes."""
    Process = _NoDaemonProcess

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
            #print(f"[INFO] Detected {vram_gb:.1f}GB VRAM per GPU. Running 122B model on 1 GPU.")
            #return 1
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
    
    # DeepSeek might still need the manual injection, but we remove Qwen from here
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

    # Setup specific kwargs for the chat template
    # If it's a Qwen model, pass enable_thinking=True to activate native reasoning tags
    template_kwargs = {"add_generation_prompt": True}
    if "qwen" in model_name.lower():
        template_kwargs["enable_thinking"] = True  # Set to False if you want to strictly disable thinking
        
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

def prompt(id, cuda_devices, number_of_threads, df_all, text_column_name, model_name, max_model_len, template, output_column_name, gpu_memory_utilization, tensor_parallel_size):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    df_thread = [df_all.iloc[x:x+math.ceil(len(df_all)/number_of_threads)] for x in list(range(len(df_all)))[::math.ceil(len(df_all)/number_of_threads)]][id].copy()
    texts = list(df_thread[text_column_name])
    
    llm, tokenizer, actual_max_len = get_llm_and_tokenizer(model_name, gpu_memory_utilization, tensor_parallel_size, max_model_len)
    
    prompts = [get_prompt(tokenizer, text, template, actual_max_len, model_name) for text in tqdm(texts)]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=actual_max_len))
    
    # Sauberes Trennen von Output und Reasoning
    final_outputs = []
    reasoning_outputs = []
    
    for x in outputs:
        raw_text = x.outputs[0].text
        # Falls das Modell eine andere Konvention verwendet hat, normalisieren wir das:
        raw_text = raw_text.replace("assistantfinal", "</think>")
        
        if "</think>" in raw_text:
            parts = raw_text.split("</think>")
            reasoning = parts[0].replace("<think>", "").strip()
            final_answer = parts[1].replace("```json", "").replace("```", "").strip()
        else:
            reasoning = ""
            final_answer = raw_text.replace("```json", "").replace("```", "").strip()
            
        final_outputs.append(final_answer)
        reasoning_outputs.append(reasoning)

    # In den Dataframe schreiben
    df_thread[output_column_name] = final_outputs
    df_thread[f"{output_column_name}_reasoning"] = reasoning_outputs
    
    return df_thread

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', type=str, help='model to be used', default='gemma-3-27b')
    parser.add_argument('--dataset', nargs='?', type=str, help='dataset to be used', default='SinclairSchneider/eu_vs_disinfo')
    parser.add_argument('--text_column', nargs='?', type=str, help='name of the text column of the dataset', default='summary')
    parser.add_argument('--gpus', nargs='?', type=int, help='number of GPUs, default 4', default=4)
    parser.add_argument('--max_model_len', nargs='?', type=int, help='max model length, default 8192', default=8192)
    parser.add_argument('--output_column_name', nargs='?', type=str, help='name of the output column to be created. Default model name', default='')
    parser.add_argument('--prompt_file_name', nargs='?', type=str, help='name of the file containing the prompt template. Default prompt.txt', default='prompt.txt')
    parser.add_argument('--gpu_memory_utilization', nargs='?', type=float, help='Value between 0.0 and 1.0 for GPU usage', default=0.0)
    parser.add_argument('--testing', action='store_true', help='use just 1%% of the dataset for testing')

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    total_gpus = args.gpus
    nameTextColumn = args.text_column
    output_column_name = args.output_column_name if args.output_column_name != "" else model_name.split("/")[-1]
    prompt_file_name = args.prompt_file_name
    testing = args.testing
    max_model_len = args.max_model_len
    gpu_memory_utilization = args.gpu_memory_utilization
    
    tensor_parallel_size = get_tensor_parallel_size(model_name)
    numberOfThreads = total_gpus // tensor_parallel_size
    
    if numberOfThreads == 0:
        raise ValueError(f"Nicht genügend GPUs. Das Modell '{model_name}' benötigt {tensor_parallel_size} GPUs pro Instanz, aber es wurden nur {total_gpus} angegeben.")

    # Fetch available GPUs from the environment, otherwise default to a basic range
    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env_gpus:
        gpu_pool = env_gpus.split(",")
    else:
        gpu_pool = [str(x) for x in range(total_gpus)]
        
    cuda_devices_list = [",".join(gpu_pool[i * tensor_parallel_size : (i + 1) * tensor_parallel_size]) for i in range(numberOfThreads)]
    
    if ".json" in dataset_name:
        df = pd.read_json(dataset_name)
        if testing:
            df = df.head(int(len(df)*0.01))
        df["index"] = range(len(df))
    else:
        ds = load_dataset(dataset_name, split="train")
        if testing:
            ds = ds.train_test_split(test_size=0.01, seed=42)["test"]
        
        ds = ds.add_column("index", list(range(len(ds))))
        df = ds.to_pandas()

    template = ""
    if not os.path.isfile(prompt_file_name):
        raise Exception("Prompt file: "+prompt_file_name+" doesn't exist")

    with open(prompt_file_name, "r") as f:
        template = f.read()
    
    ldf = [df]*numberOfThreads
    lid = list(range(numberOfThreads))
    lcuda_devices = cuda_devices_list
    lNumberOfThreads = [numberOfThreads]*numberOfThreads
    lnameTextColumn = [nameTextColumn]*numberOfThreads
    lmodel_name = [model_name]*numberOfThreads
    lmax_model_len = [max_model_len]*numberOfThreads
    loutput_column_name = [output_column_name]*numberOfThreads
    ltemplate = [template]*numberOfThreads
    lgpu_memory_utilization = [gpu_memory_utilization]*numberOfThreads
    ltensor_parallel_size = [tensor_parallel_size]*numberOfThreads
    
    lArguments = list(zip(lid, lcuda_devices, lNumberOfThreads, ldf, lnameTextColumn, lmodel_name, lmax_model_len, ltemplate, loutput_column_name, lgpu_memory_utilization, ltensor_parallel_size))

    with NonDaemonPool(processes=numberOfThreads) as pool:
        result = pool.starmap(prompt, lArguments)
        df_result = pd.concat(result)
        df_result.set_index('index', inplace=True)
        df_result.sort_index(inplace=True)
        output_name = dataset_name.split("/")[-1].replace(".json","")+"_BY_"+output_column_name+".json"
        df_result.to_json(output_name)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
