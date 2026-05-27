from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import uvicorn
import argparse
import threading
import os
import json
import time

app = FastAPI()

# Global State
lock = threading.Lock()
dataset_df = None
unprocessed_indices = []
processing_batches = {}  # Map: task_id -> {"indices": [1,2,3], "dispatched_at": timestamp}
results_data = {}        # Map: index -> result dict
output_filename = ""
pbar = None              # Global progress bar
task_counter = 0         # Running number for Task IDs

# Configuration set by Master to hand out to clients
master_config = {}

class BatchResult(BaseModel):
    index: int
    output: str
    reasoning: str

class SubmitRequest(BaseModel):
    batch_id: str
    results: list[BatchResult]

def timeout_checker():
    """Background thread that monitors for timed-out tasks and re-queues them."""
    global unprocessed_indices  # Must be declared global before use
    
    # Wait until master_config is populated before trying to read from it
    while "task_timeout_seconds" not in master_config:
        time.sleep(1)
        
    timeout_limit = master_config.get("task_timeout_seconds", 3600)
    while True:
        time.sleep(30) # Check every 30 seconds
        with lock:
            # If both queues are completely empty, the job is done
            if not unprocessed_indices and not processing_batches:
                break 
            
            now = time.time()
            expired_tasks = []
            
            for task_id, data in processing_batches.items():
                if now - data["dispatched_at"] > timeout_limit:
                    expired_tasks.append(task_id)
            
            for task_id in expired_tasks:
                print(f"\n[MASTER] Task {task_id} timed out (> {timeout_limit}s). Re-queuing {len(processing_batches[task_id]['indices'])} items...")
                # Prepend the indices so they get picked up immediately
                unprocessed_indices = processing_batches[task_id]["indices"] + unprocessed_indices
                del processing_batches[task_id]

@app.get("/get_batch")
def get_batch():
    global task_counter
    with lock:
        if not unprocessed_indices:
            return {"batch_id": None, "data": [], "config": None}
        
        size = master_config["batch_size"]
        batch_indices = unprocessed_indices[:size]
        del unprocessed_indices[:size]
        
        # Create a new running task ID
        task_counter += 1
        task_id = str(task_counter)
        
        processing_batches[task_id] = {
            "indices": batch_indices,
            "dispatched_at": time.time()
        }
        
        data = [{"index": idx, "text": dataset_df.loc[idx, app.state.text_column]} for idx in batch_indices]
        
        return {
            "batch_id": task_id, 
            "data": data,
            "config": master_config
        }

@app.post("/submit_batch")
def submit_batch(req: SubmitRequest):
    with lock:
        # If task timed out and was re-queued, it won't be in processing_batches anymore
        if req.batch_id not in processing_batches:
            print(f"\n[MASTER] Rejected late submission for Task {req.batch_id}")
            raise HTTPException(status_code=400, detail="Task rejected: Expired and reassigned.")
        
        # Store results
        for res in req.results:
            results_data[res.index] = {
                "output": res.output,
                "reasoning": res.reasoning
            }
        
        if pbar:
            pbar.update(len(req.results))
        
        # Remove from processing queue
        del processing_batches[req.batch_id]

        if not unprocessed_indices and not processing_batches:
            if pbar:
                pbar.close()
            print("\n[MASTER] All batches processed! Saving to disk...")
            save_results()

    return {"status": "success"}

def save_results():
    global dataset_df
    outputs = []
    reasonings = []
    
    for idx in range(len(dataset_df)):
        res = results_data.get(idx, {"output": "", "reasoning": ""})
        outputs.append(res["output"])
        reasonings.append(res["reasoning"])
        
    dataset_df[app.state.output_column_name] = outputs
    dataset_df[f"{app.state.output_column_name}_reasoning"] = reasonings
    
    dataset_df.to_json(output_filename, orient="records")
    print(f"[MASTER] Successfully saved to {output_filename}")
    os._exit(0)

def generate_config(filename="config.json"):
    config = {
        "dataset": "SinclairSchneider/eu_vs_disinfo",
        "text_column": "summary",
        "output_column_name": "",
        "port": 8192,
        "batch_size": 1000,
        "model": "gemma-3-27b",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.0,
        "prompt_file_name": "prompt.txt",
        "task_timeout_seconds": 3600
    }
    with open(filename, "w") as f:
        f.write(json.dumps(config, indent=4))
    print(f"[MASTER] {filename} generated successfully.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_config', nargs='?', const='config.json', default=None, help='generates an empty config file (optionally provide a filename)')
    parser.add_argument('--config', nargs='?', type=str, help='config json-file', default='config.json')
    parser.add_argument('--testing', action='store_true', help='use just 1%% of the dataset for testing')
    
    args = parser.parse_args()

    # Trigger generation if flag is present
    if args.generate_config is not None:
        generate_config(args.generate_config)
        return

    config_file = args.config
    if not os.path.isfile(config_file):
        print(f"[ERROR] Config file '{config_file}' not found. Run with --generate_config to create one.")
        return

    with open(config_file, "r") as f:
        config = json.loads(f.read())

    global dataset_df, unprocessed_indices, output_filename, pbar, master_config

    prompt_file_name = config.get("prompt_file_name", "prompt.txt")
    if not os.path.isfile(prompt_file_name):
        raise Exception(f"Prompt file: '{prompt_file_name}' doesn't exist")
        
    with open(prompt_file_name, "r") as f:
        prompt_template = f.read()

    model_name = config.get("model", "gemma-3-27b")
    master_config = {
        "batch_size": config.get("batch_size", 1000),
        "model": model_name,
        "max_model_len": config.get("max_model_len", 8192),
        "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.0),
        "template": prompt_template,
        "task_timeout_seconds": config.get("task_timeout_seconds", 3600)
    }

    dataset_name = config.get("dataset", "SinclairSchneider/eu_vs_disinfo")
    print(f"[MASTER] Loading dataset: {dataset_name}...")
    
    if ".json" in dataset_name:
        dataset_df = pd.read_json(dataset_name)
        if args.testing:
            dataset_df = dataset_df.head(int(len(dataset_df)*0.01))
    else:
        ds = load_dataset(dataset_name, split="train")
        if args.testing:
            ds = ds.train_test_split(test_size=0.01, seed=42)["test"]
        dataset_df = ds.to_pandas()
    
    dataset_df = dataset_df.reset_index(drop=True)
    unprocessed_indices = list(range(len(dataset_df)))
    
    config_out_col = config.get("output_column_name", "")
    output_col = config_out_col if config_out_col != "" else model_name.split("/")[-1]
    
    app.state.text_column = config.get("text_column", "summary")
    app.state.output_column_name = output_col
    output_filename = dataset_name.split("/")[-1].replace(".json", "") + f"_BY_{output_col}.json"
    
    print(f"[MASTER] Target output file: {output_filename}")
    
    pbar = tqdm(total=len(dataset_df), desc="Processing Dataset", unit="item")
    
    # Start the timeout background monitor
    threading.Thread(target=timeout_checker, daemon=True).start()
    
    port = config.get("port", 8192)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

if __name__ == '__main__':
    main()