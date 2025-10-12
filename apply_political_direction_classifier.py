import os
import multiprocessing
from datasets import load_dataset
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import math
from tqdm import tqdm
import os
import torch
import pandas as pd
import argparse
import numpy as np

def classify(id, numberOfThreads, df_all, nameTextColumn, batchSize, model_name, max_position_embeddings):
    if "german_politic_direction_gemma-2-2b" in model_name.lower():
        vectors = np.array([[-1, 0],
                    [-9.99193435e-01,  4.01556900e-02],
                    [-6.85641955e-01,  7.27938947e-01],
                    [ 3.82683432e-01,  9.23879533e-01],
                    [ 8.69790824e-01,  4.93420634e-01],
                    [1, 0]])
    elif "german_politic_direction_gemma-2-9b" in model_name.lower():
        vectors = np.array([[-1, 0], 
                    [-9.99193435e-01,  4.01556900e-02], 
                    [-7.91445449e-01,  6.11239806e-01], 
                    [ 3.82683432e-01,  9.23879533e-01], 
                    [ 8.69790824e-01,  4.93420634e-01], 
                    [1, 0]])

    #print("id: "+str(id))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True, torch_dtype = torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, TOKENIZERS_PARALLELISM=True, trust_remote_code=True, max_length=max_position_embeddings, truncation=True)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, trust_remote_code=True, device=id, torch_dtype = torch.bfloat16, max_length=max_position_embeddings, truncation=True)
    
    df_thread = [df_all.iloc[x:x+math.ceil(len(df_all)/numberOfThreads)] for x in list(range(len(df_all)))[::math.ceil(len(df_all)/numberOfThreads)]][id]
    batches = [df_thread.iloc[x:x+batchSize] for x in list(range(len(df_thread)))[::batchSize]]

    extraColumns = {}
    results = []
    for batch in tqdm(batches):
        with torch.no_grad():
            batch_input = [x if x != None else '' for x in batch[nameTextColumn]]
            batch_input = [" ".join(x.split(" ")[:max_position_embeddings]) for x in list(batch_input)]
            classification_results = np.array([pd.DataFrame(x).sort_values(by=['label'], key=lambda x: x.map({'DIE LINKE':0, 'BÜNDNIS 90/DIE GRÜNEN':1, 'SPD':2, 'FDP':3, 'CDU/CSU':4, 'AfD':5}))['score'] for x in pipe(batch_input, top_k=None, batch_size=batchSize)])
            classification_results = [np.arctan2(*x)/(np.pi/2) for x in classification_results@vectors]
            results.extend(classification_results)
        #torch.cuda.empty_cache()
        #for result in results:
        #    #for result_label in result:
        #    #    if result_label['label'] not in extraColumns:
        #    #        extraColumns[result_label['label']] = []
        #    #    extraColumns[result_label['label']].append(result_label['score'])

    df_thread_return = df_thread.copy()
    df_thread_return["politic_left_right"] = results
    
    return df_thread_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='?', type=str, help='model to be used, either SinclairSchneider/german_politic_direction_gemma-2-2b or SinclairSchneider/german_politic_direction_gemma-2-9b (default)', default='SinclairSchneider/german_politic_direction_gemma-2-9b')
    parser.add_argument('--dataset', nargs='?', type=str, help='dataset (huggingface or pandas-json) to be used, default SinclairSchneider/tweets_about_german_politicians_jan_feb_2025', default='SinclairSchneider/tweets_about_german_politicians_jan_feb_2025')
    parser.add_argument('--text_column', nargs='?', type=str, help='name of the text column of the dataset, default text', default='text')
    parser.add_argument('--gpus', nargs='?', type=int, help='number of GPUs, default 4', default=4)
    parser.add_argument('--batch_size', nargs='?', type=int, help='batch size, default 4', default=4)
    parser.add_argument('--max_position_embeddings', nargs='?', type=int, help='max position embeddings (max input of the model), default 8192', default=8192)
    parser.add_argument('--testing', action='store_true', help='use just 1%% of the dataset for testing') 

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset
    numberOfThreads = args.gpus
    batchSize = args.batch_size
    nameTextColumn = args.text_column
    testing = args.testing
    max_position_embeddings = args.max_position_embeddings
    
    if model_name.lower() != "sinclairschneider/german_politic_direction_gemma-2-2b" and model_name.lower() != "sinclairschneider/german_politic_direction_gemma-2-9b":
        print("Model must be either SinclairSchneider/german_politic_direction_gemma-2-2b or SinclairSchneider/german_politic_direction_gemma-2-9b")
        return

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
    
    ldf = [df]*numberOfThreads
    lid = list(range(numberOfThreads))
    lNumberOfThreads = [numberOfThreads]*numberOfThreads
    lnameTextColumn = [nameTextColumn]*numberOfThreads
    lbatchSize = [batchSize]*numberOfThreads
    lmodel_name = [model_name]*numberOfThreads
    lmax_position_embeddings = [max_position_embeddings]*numberOfThreads
    lArguments = list(zip(lid, lNumberOfThreads, ldf, lnameTextColumn, lbatchSize, lmodel_name, lmax_position_embeddings))

    with multiprocessing.Pool(processes=numberOfThreads) as pool:
        result = pool.starmap(classify, lArguments)
        df_result = pd.concat(result)
        df_result.set_index('index', inplace=True)
        df_result.sort_index(inplace=True)
        #output_name = dataset_name.split("/")[-1].replace(".json","")+"_politic_left_right_BY_"+model_name.split("/")[-1]+".json"
        if "gemma-2-2b" in model_name.lower():
            output_name = dataset_name.split("/")[-1].replace(".json","")+"_politic_left_right_BY_gemma-2-2b.json"
        elif "gemma-2-9b" in model_name.lower():
            output_name = dataset_name.split("/")[-1].replace(".json","")+"_politic_left_right_BY_gemma-2-9b.json"
        df_result.to_json(output_name)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
