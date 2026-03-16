import multiprocessing as mp
import multiprocessing.pool as mpp
import ctypes
import signal
import argparse
import os
import math
import pandas as pd
import torch
import numpy as np
from vllm import LLM
from datasets import load_dataset
from tqdm import tqdm

class _NoDaemonProcess(mp.Process):
    """A Process subclass that is *not* daemonic **and** is compatible with
    the way `multiprocessing.pool` constructs workers since Python 3.12."""
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

def embed_partition(id, cuda_devices, number_of_threads, df_all, text_column_name, model_name, output_column_name, gpu_memory_utilization, tensor_parallel_size, instruction):
    """Worker function to process a chunk of the dataframe on a specific GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    
    # Partition the dataframe evenly across available threads
    chunk_size = math.ceil(len(df_all) / number_of_threads)
    start_idx = id * chunk_size
    end_idx = start_idx + chunk_size
    df_thread = df_all.iloc[start_idx:end_idx].copy()
    
    texts = list(df_thread[text_column_name])
    texts = [str(x) for x in texts]
    
    # Prepend the dynamically passed instruction
    texts = [instruction + x for x in texts]
    
    # Initialize vLLM with the pooling runner for embeddings
    llm_kwargs = {
        "model": model_name,
        "trust_remote_code": True,
        "runner": "pooling", 
        "tensor_parallel_size": tensor_parallel_size
    }
    
    if gpu_memory_utilization > 0.0:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
        
    print(f"[Thread {id}] Loading embedding model {model_name} on GPU(s) {cuda_devices}...")
    model = LLM(**llm_kwargs)
    
    # Generate embeddings
    print(f"[Thread {id}] Processing {len(texts)} texts...")
    outputs = model.embed(texts)
    
    # Extract and normalize embeddings (L2 norm)
    embeddings = torch.tensor([x.outputs.embedding for x in outputs], dtype=torch.float32).numpy()
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1e-12, norm) # Avoid division by zero safely
    embeddings = (embeddings / norm).tolist()
    
    # Write back to the thread's dataframe chunk
    df_thread[output_column_name] = embeddings
    
    return df_thread

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', type=str, help='Model to be used', default='Qwen/Qwen3-Embedding-8B')
    parser.add_argument('--dataset', nargs='?', type=str, help='Dataset to be used (HF dataset name, .json, or .parquet)', required=True)
    parser.add_argument('--text_column', nargs='?', type=str, help='Name of the text column of the dataset', default='text')
    parser.add_argument('--gpus', nargs='?', type=int, help='Total number of GPUs to use', default=4)
    parser.add_argument('--tensor_parallel_size', nargs='?', type=int, help='GPUs required per model instance', default=1)
    parser.add_argument('--output_column_name', nargs='?', type=str, help='Name of the output column', default='embeddings')
    parser.add_argument('--gpu_memory_utilization', nargs='?', type=float, help='Value between 0.0 and 1.0 for GPU usage', default=0.0)
    parser.add_argument('--instruction', nargs='?', type=str, help='Instruction prefix for the embedding model', default='Identify the main topic of the following text: ')
    parser.add_argument('--testing', action='store_true', help='Use just 1%% of the dataset for testing')

    args = parser.parse_args()
    
    # Setup GPU distribution
    number_of_threads = args.gpus // args.tensor_parallel_size
    if number_of_threads == 0:
        raise ValueError(f"Not enough GPUs. The model requires {args.tensor_parallel_size} GPUs, but only {args.gpus} provided.")

    env_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env_gpus:
        gpu_pool = env_gpus.split(",")
    else:
        gpu_pool = [str(x) for x in range(args.gpus)]
        
    cuda_devices_list = [",".join(gpu_pool[i * args.tensor_parallel_size : (i + 1) * args.tensor_parallel_size]) for i in range(number_of_threads)]
    
    # Load Dataset
    print(f"[INFO] Loading dataset: {args.dataset}")
    if ".json" in args.dataset:
        df = pd.read_json(args.dataset)
    elif ".parquet" in args.dataset:
        df = pd.read_parquet(args.dataset)
    else:
        ds = load_dataset(args.dataset, split="train")
        df = ds.to_pandas()

    if args.testing:
        print("[INFO] Testing mode enabled: Using 1% of the dataset.")
        df = df.head(max(1, int(len(df) * 0.01)))
        
    df["index"] = range(len(df))

    # Setup multiprocessing arguments mapping
    ldf = [df] * number_of_threads
    lid = list(range(number_of_threads))
    lcuda_devices = cuda_devices_list
    lNumberOfThreads = [number_of_threads] * number_of_threads
    lnameTextColumn = [args.text_column] * number_of_threads
    lmodel_name = [args.model] * number_of_threads
    loutput_column_name = [args.output_column_name] * number_of_threads
    lgpu_memory_utilization = [args.gpu_memory_utilization] * number_of_threads
    ltensor_parallel_size = [args.tensor_parallel_size] * number_of_threads
    linstruction = [args.instruction] * number_of_threads
    
    lArguments = list(zip(
        lid, lcuda_devices, lNumberOfThreads, ldf, 
        lnameTextColumn, lmodel_name, loutput_column_name, 
        lgpu_memory_utilization, ltensor_parallel_size, linstruction
    ))

    print(f"[INFO] Starting {number_of_threads} parallel workers...")
    with NonDaemonPool(processes=number_of_threads) as pool:
        results = pool.starmap(embed_partition, lArguments)
        
        # Reconstruct final dataframe
        df_result = pd.concat(results)
        df_result.set_index('index', inplace=True)
        df_result.sort_index(inplace=True)
        
        # --- NEW: Sanitize columns for Parquet strict typing ---
        print("[INFO] Sanitizing mixed-type columns for Parquet compatibility...")
        for col in df_result.columns:
            # Convert object columns to strings (except our new embeddings column, which is purely lists of floats)
            if df_result[col].dtype == 'object' and col != args.output_column_name:
                df_result[col] = df_result[col].astype(str)
        # -------------------------------------------------------

        # Save to Parquet
        base_name = args.dataset.split("/")[-1].replace(".json", "").replace(".parquet", "")
        output_name = f"{base_name}_BY_{args.output_column_name}.parquet"
        
        print(f"[INFO] Saving results to {output_name}...")
        df_result.to_parquet(output_name, index=False, engine="pyarrow", row_group_size=10000, compression="snappy")
        print("[SUCCESS] Done!")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
