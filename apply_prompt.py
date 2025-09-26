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
    the way `multiprocessing.pool` constructs workers since Python 3.12.

    The pool implementation calls `Process(ctx, …)` where the **first** positional
    argument is the *context* object, *not* the traditional `group` parameter.
    We therefore strip that extra leading argument off before delegating to the
    real `multiprocessing.Process.__init__`.
    """

    def __init__(self, *args, **kwargs):
        # If the first positional arg is a BaseContext instance (spawn/fork ctx),
        # drop it so that the remaining args match the usual (group, target, …)
        if args and isinstance(args[0], mp.context.BaseContext):
            args = args[1:]
        super().__init__(*args, **kwargs)

        # Tell kernel to send SIGKILL if parent dies
        libc = ctypes.CDLL(None)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)

    # Force daemon=False regardless of what Pool tries to set
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):  # ignore attempts to set
        pass


class NonDaemonPool(mpp.Pool):
    """A `multiprocessing.Pool` whose workers may spawn child processes."""

    Process = _NoDaemonProcess
    """A multiprocessing Pool whose workers can spawn children."""

def get_llm_and_tokenizer(model_name, gpu_memory_utilization):
    model_name_hf = ""
    model_name = model_name.lower()
    if model_name == "gemma-3-27b":
        model_name_hf  = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
        tokenizer = AutoProcessor.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name == "llama-3.3-70b":
        model_name_hf = "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name == "qwen3-30b":
        model_name_hf = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name == "qwen3-32b":
        model_name_hf = "RedHatAI/Qwen3-32B-FP8-dynamic"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name == "deepseek-r1-70b":
        model_name_hf = "RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name == "gpt-oss-20b":
        model_name_hf = "openai/gpt-oss-20b"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    elif model_name == "glm-z1-32b":
        model_name_hf = "duydq12/GLM-Z1-32B-0414-FP8-dynamic"
        tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=True)
    else:
        raise Exception("Please chose one of the models: gemma-3-27b, llama-3.3-70b, qwen3-30b, qwen3-32b, deepseek-r1-70b, gpt-oss-20b")

    if gpu_memory_utilization == 0.0:
        llm = LLM(model=model_name_hf, trust_remote_code=True, max_model_len=8192)
    else:
        llm = LLM(model=model_name_hf, trust_remote_code=True, max_model_len=8192, gpu_memory_utilization=gpu_memory_utilization)
    
    return llm, tokenizer

def get_prompt(tokenizer, text, template):
    text = text if text is not None else ""
    prompt_text = template + text
    if "gemma" in str(type(tokenizer)):
        chat = [
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
            {"role": "assistant", "content": []}
        ]
    else:
        chat = [
            {"role": "user", "content": prompt_text,},
        ]
        
    return chat

def prompt(id, number_of_threads, df_all, text_column_name, model_name, max_model_len, template, output_column_name, gpu_memory_utilization):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
    torch.cuda.set_device(0)
    
    df_thread = [df_all.iloc[x:x+math.ceil(len(df_all)/number_of_threads)] for x in list(range(len(df_all)))[::math.ceil(len(df_all)/number_of_threads)]][id].copy()
    texts = list(df_thread[text_column_name])
    llm, tokenizer = get_llm_and_tokenizer(model_name, gpu_memory_utilization)
    prompts = tokenizer.apply_chat_template([get_prompt(tokenizer, text, template) for text in texts], tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(prompts, SamplingParams(temperature=0.8, max_tokens=max_model_len))
    output_texts = [x.outputs[0].text.replace("```json", "").replace("```", "").strip() if "</think>" not in x.outputs[0].text.replace("assistantfinal", "</think>") \
                    else x.outputs[0].text.replace("assistantfinal", "</think>").split("</think>")[1].replace("```json", "").replace("```", "").strip() for x in outputs]

    df_thread[output_column_name] = output_texts
    
    return df_thread

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', type=str, help='model to be used, default gemma-3-27b. Possible models: gemma-3-27b, llama-3.3-70b, qwen3-30b, qwen3-32b, deepseek-r1-70b, gpt-oss-20b, glm-z1-32b', default='gemma-3-27b')
    parser.add_argument('--dataset', nargs='?', type=str, help='dataset (huggingface or pandas-json) to be used, default SinclairSchneider/eu_vs_disinfo', default='SinclairSchneider/eu_vs_disinfo')
    parser.add_argument('--text_column', nargs='?', type=str, help='name of the text column of the dataset, default summary', default='summary')
    parser.add_argument('--gpus', nargs='?', type=int, help='number of GPUs, default 4', default=4)
    parser.add_argument('--max_model_len', nargs='?', type=int, help='max model lengeth, default 8192', default=8192)
    parser.add_argument('--output_column_name', nargs='?', type=str, help='name of the output column to be created. Default model name', default='')
    parser.add_argument('--prompt_file_name', nargs='?', type=str, help='name of the file containing the prompt template. Default prompt.txt', default='prompt.txt')
    parser.add_argument('--gpu_memory_utilization', nargs='?', type=float, help='Value between 0.0 and 1.0 for GPU usage', default=0.0)
    parser.add_argument('--testing', action='store_true', help='use just 1%% of the dataset for testing')

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    numberOfThreads = args.gpus
    nameTextColumn = args.text_column
    output_column_name = args.output_column_name if args.output_column_name != "" else model_name.split("/")[-1]
    prompt_file_name = args.prompt_file_name
    testing = args.testing
    max_model_len = args.max_model_len
    gpu_memory_utilization = args.gpu_memory_utilization
    
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
    lNumberOfThreads = [numberOfThreads]*numberOfThreads
    lnameTextColumn = [nameTextColumn]*numberOfThreads
    lmodel_name = [model_name]*numberOfThreads
    lmax_model_len = [max_model_len]*numberOfThreads
    loutput_column_name = [output_column_name]*numberOfThreads
    ltemplate = [template]*numberOfThreads
    lgpu_memory_utilization = [gpu_memory_utilization]*numberOfThreads
    
    lArguments = list(zip(lid, lNumberOfThreads, ldf, lnameTextColumn, lmodel_name, lmax_model_len, ltemplate, loutput_column_name, lgpu_memory_utilization))

    #with multiprocessing.Pool(processes=numberOfThreads) as pool:
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
