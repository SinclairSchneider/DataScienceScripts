import multiprocessing as mp
from multiprocessing import process
from sqlalchemy import create_engine
import argparse
import json
import os
import pandas as pd
import multiprocessing.pool as mpp
import ctypes, signal
import torch
import numpy as np
from vllm import LLM
from tqdm import tqdm

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

def generate_config():
    config = {
        'model': 'Qwen/Qwen3-Embedding-8B',
        'desired_batch_size': 100000,
        'number_of_threads': 4,
        'db_database_name': 'default',
        'db_username': 'default',
        'db_password': 'password',
        'db_host': '127.0.0.1',
        'db_port': 9000,
        'db_input_table': 'input_table',
        'db_input_text_column': 'text',
        'db_input_hash_column': 'id',
        'db_output_table': 'output_table',
        'db_output_embeddings_column': 'embeddings',
    }
    with open("config.json", "w") as f:
        f.write(json.dumps(config, indent=1))

def get_batch(engine, db_database_name, db_input_table, db_input_hash_column, number_of_threads, thread_id, batches_per_thread, batch_id, limit):
    sql_statement = "SELECT * \
                     FROM "+db_database_name+"."+db_input_table+" \
                     WHERE cityHash64("+db_input_hash_column+")%%"+str(number_of_threads)+"="+str(thread_id)+" \
                     and cityHash64(cityHash64("+db_input_hash_column+"))%%"+str(batches_per_thread)+"="+str(batch_id)
    if limit > 0:
        sql_statement = sql_statement + " LIMIT "+str(limit)
    df = pd.read_sql(sql_statement, con=engine)
    return df

def process_thread(thread_id, number_of_threads, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(thread_id)
    torch.cuda.set_device(0)

    testing = config.get("testing", False)
    desired_batch_size = config.get("desired_batch_size", 100000)
    model_name = config.get("model", "Qwen/Qwen3-Embedding-8B")
    db_database_name = config.get("db_database_name", "default")
    db_username = config.get("db_username", "default")
    db_password = config.get("db_password", "")
    db_host = config.get("db_host", "127.0.0.1")
    db_port = config.get("db_port", 9000)

    db_input_table = config.get("db_input_table", "")
    db_input_text_column = config.get("db_input_text_column", "text")
    db_input_hash_column = config.get("db_input_hash_column", "id")
    db_output_table = config.get("db_output_table", "")
    db_output_embeddings_column = config.get("db_output_embeddings_column", "embeddings")
    batches_per_thread = config.get("batches_per_thread", 100)
    gpu_memory_utilization = config.get("gpu_memory_utilization", 0.0)

    engine = create_engine("clickhouse+native://"+db_username+":"+db_password+"@"+db_host+":"+str(db_port)+"/"+db_database_name)
    if testing:
        limit = int(desired_batch_size*0.01)
    else:
        limit = 0

    if gpu_memory_utilization == 0.0:
        model = LLM(model=model_name, task="embed", trust_remote_code=True)
    else:
        model = LLM(model=model_name, task="embed", trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization)
    
    for batch_id in tqdm(range(batches_per_thread)):
        df = get_batch(engine, db_database_name, db_input_table, db_input_hash_column, number_of_threads, thread_id, batches_per_thread, batch_id, limit)
        texts = list(df[db_input_text_column])
        texts = [str(x) for x in texts]
        outputs = model.embed(texts)

        embeddings = torch.tensor([x.outputs.embedding for x in outputs], dtype=torch.float32).numpy()
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm = np.matrix([x[0] if x[0]!=0 else np.float32(1e-12) for x in norm]).T
        embeddings = embeddings/norm
        embeddings = [[float(x) for x in list(x)] for x in list(np.array(embeddings))]

        df[db_output_embeddings_column] = embeddings
        df.to_sql(db_output_table, con=engine, if_exists="append", index=False)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_config', action="store_true", help='generates an empty config file')
    parser.add_argument('--config', nargs='?', type=str, help='config json-file that has been created and adjusted', default='config.json')
    parser.add_argument('--testing', action='store_true', help='use just 1%% of the dataset for testing') 
    parser.add_argument('--gpu_memory_utilization', nargs='?', type=float, help='Value between 0.0 and 1.0 for GPU usage', default=0.0)
    
    args = parser.parse_args()
    config_file = args.config
    testing = args.testing
    gpu_memory_utilization = args.gpu_memory_utilization

    if args.generate_config:
        generate_config()
        return

    if not os.path.isfile(config_file):
        print("Config file not found")
        return
    
    with open(config_file, "r") as f:
        config = json.loads(f.read())

    model = config.get("model", "Qwen/Qwen3-Embedding-8B")
    desired_batch_size = config.get("desired_batch_size", 100000)
    number_of_threads = config.get("number_of_threads", 1)
    db_database_name = config.get("db_database_name", "default")
    db_username = config.get("db_username", "default")
    db_password = config.get("db_password", "")
    db_host = config.get("db_host", "127.0.0.1")
    db_port = config.get("db_port", 9000)
    db_input_table = config.get("db_input_table", "")
    db_input_text_column = config.get("db_input_text_column", "text")
    db_input_hash_column = config.get("db_input_hash_column", "id")
    db_output_table = config.get("db_output_table", "")
    db_output_embeddings_column = config.get("db_output_embeddings_column", "embeddings")
    
    engine = create_engine("clickhouse+native://"+db_username+":"+db_password+"@"+db_host+":"+str(db_port)+"/"+db_database_name)
    df = pd.read_sql("SELECT toInt64(count(*)/"+str(desired_batch_size)+")+1 as divisor \
                  FROM "+db_database_name+"."+db_input_table+" \
                  WHERE cityHash64("+db_input_hash_column+")%%"+str(number_of_threads)+"=0", con=engine)
    batches_per_thread = int(df['divisor'][0])

    config['testing'] = testing
    config['batches_per_thread'] = batches_per_thread
    config['gpu_memory_utilization'] = gpu_memory_utilization
    
    lid = list(range(number_of_threads))
    lnumber_of_threads = [number_of_threads]*number_of_threads
    lconfig = [config]*number_of_threads

    lArguments = list(zip(lid, lnumber_of_threads, lconfig))
    with NonDaemonPool(processes=number_of_threads) as pool:
        pool.starmap(process_thread, lArguments)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
