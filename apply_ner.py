import os
import multiprocessing
from datasets import load_dataset
import math
from tqdm import tqdm
import os
import torch
import pandas as pd
import argparse
import spacy
from ftlangdetect import detect

#https://github.com/LlmKira/fast-langdetect
#nameTextColumn = "content"
#numberOfThreads = 4
#batchSize = 4
#model_name = "SinclairSchneider/german_politic_EuroBERT-210m"
#dataset_name = "SinclairSchneider/deutschlandfunk_de"

def ner(id, numberOfThreads, df_all, nameTextColumn):
    spacy.require_gpu(id)
    df_thread = [df_all.iloc[x:x+math.ceil(len(df_all)/numberOfThreads)] for x in list(range(len(df_all)))[::math.ceil(len(df_all)/numberOfThreads)]][id].copy()
    batches = [df_thread.iloc[x:x+1] for x in range(len(df_thread))]
    
    nlp_dict = {}
    result_dict = {}
    l_lang = []
    for batch in tqdm(batches):
        batch_input = list(batch[nameTextColumn])[0]
        batch_input = batch_input if batch_input != None else ""
        lang = detect(text=batch_input.replace("\n", ""), low_memory=True).get('lang', '')
        l_lang.append(lang)
    
        if lang in nlp_dict:
            nlp = nlp_dict[lang]
        else:
            if lang == 'en':
                nlp = spacy.load("en_core_web_trf")
                nlp_dict[lang] = nlp
            elif lang == 'de':
                nlp = spacy.load("de_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'ca':
                nlp = spacy.load("ca_core_news_trf")
                nlp_dict[lang] = nlp
            elif lang == 'zh':
                nlp = spacy.load("zh_core_web_trf")
                nlp_dict[lang] = nlp
            elif lang == 'hr':
                nlp = spacy.load("hr_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'da':
                nlp = spacy.load("da_core_news_trf")
                nlp_dict[lang] = nlp
            elif lang == 'nl':
                nlp = spacy.load("nl_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'fi':
                nlp = spacy.load("fi_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'fr':
                nlp = spacy.load("fr_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'el':
                nlp = spacy.load("el_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'it':
                nlp = spacy.load("it_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'ja':
                nlp = spacy.load("ja_core_news_trf")
                nlp_dict[lang] = nlp
            elif lang == 'ko':
                nlp = spacy.load("ko_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'lt':
                nlp = spacy.load("lt_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'mk':
                nlp = spacy.load("mk_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'nb':
                nlp = spacy.load("nb_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'pl':
                nlp = spacy.load("pl_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'pt':
                nlp = spacy.load("pt_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'ro':
                nlp = spacy.load("ro_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'ru':
                nlp = spacy.load("ru_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'sl':
                nlp = spacy.load("sl_core_news_trf")
                nlp_dict[lang] = nlp
            elif lang == 'es':
                nlp = spacy.load("es_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'es':
                nlp = spacy.load("sv_core_news_lg")
                nlp_dict[lang] = nlp
            elif lang == 'uk':
                nlp = spacy.load("uk_core_news_trf")
                nlp_dict[lang] = nlp
            else:
                lang = "xx"
                if lang in nlp_dict:
                    nlp = nlp_dict[lang]
                else:
                    nlp = spacy.load("xx_ent_wiki_sm")
                    nlp_dict[lang] = nlp
        try:   
            doc = nlp(batch_input)
            tmp_dict = {}
            for ent in doc.ents:
                if ent.label_ not in tmp_dict:
                    tmp_dict[ent.label_] = []        
                tmp_dict[ent.label_].append(ent.text)
        
            if len(result_dict) == 0:
                ref_len = 0
            else:
                ref_len = len(result_dict[list(result_dict.keys())[0]])
                
            if len(tmp_dict) > 0:
                for key in tmp_dict:
                    tmp_dict[key] = list(set(tmp_dict[key]))
                    if key not in result_dict:
                        if len(result_dict) > 0:
                            new_list = [[]]*ref_len
                            result_dict[key] = new_list
                            result_dict[key].append(tmp_dict[key])
                        else:
                            result_dict[key] = []
                            result_dict[key].append(tmp_dict[key])
                    else:
                        result_dict[key].append(tmp_dict[key])
            else:
                for key in result_dict:
                    result_dict[key].append([])
                if len(result_dict) == 0:
                    result_dict['dummy'] = []
                    result_dict['dummy'].append([])
        
            if len(result_dict) == 0:
                max_len = 0
            else:
                max_len = max([len(result_dict[x]) for x in list(result_dict.keys())])
        
            for key in result_dict.keys():
                if len(result_dict[key]) < max_len:
                    result_dict[key] = result_dict[key] + [[]] * (max_len - len(result_dict[key]))
                    
        except:
            for key in result_dict:
                result_dict[key].append([])
            if len(result_dict) == 0:
                result_dict['dummy'] = []
                result_dict['dummy'].append([])

            if len(result_dict) == 0:
                max_len = 0
            else:
                max_len = max([len(result_dict[x]) for x in list(result_dict.keys())])
        
            for key in result_dict.keys():
                if len(result_dict[key]) < max_len:
                    result_dict[key] = result_dict[key] + [[]] * (max_len - len(result_dict[key]))
            
    
    new_columns = sorted(list(result_dict.keys()))
    for new_column in new_columns:
        if new_column == 'dummy':
            continue
        df_thread['lang'] = l_lang
        df_thread[new_column] = result_dict[new_column].copy()
    return df_thread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='?', type=str, help='dataset (huggingface or pandas-json) to be used, default SinclairSchneider/deutschlandfunk_de', default='SinclairSchneider/deutschlandfunk_de')
    parser.add_argument('--text_column', nargs='?', type=str, help='name of the text column of the dataset, default content', default='content')
    parser.add_argument('--gpus', nargs='?', type=int, help='number of GPUs, default 4', default=4)
    parser.add_argument('--testing', action='store_true', help='use just 1%% of the dataset for testing') 

    args = parser.parse_args()
    dataset_name = args.dataset
    numberOfThreads = args.gpus
    nameTextColumn = args.text_column
    testing = args.testing
    
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
    lArguments = list(zip(lid, lNumberOfThreads, ldf, lnameTextColumn))

    with multiprocessing.Pool(processes=numberOfThreads) as pool:
        result = pool.starmap(ner, lArguments)

        result_dict = {}
        for df in result:
            for key in df.keys():
                if key in result_dict:
                    result_dict[key].extend(list(df[key]))
                else:
                    if len(result_dict.keys()) == 0:
                        max_len = 0
                    else:
                        max_len = max([len(result_dict[x]) for x in list(result_dict.keys())])
                    diff = max_len - len(df[key])
                    result_dict[key] = [[]]*diff + list(df[key])
                    
            max_len = max([len(result_dict[x]) for x in list(result_dict.keys())])
            for key in result_dict:
                if len(result_dict[key]) < max_len:
                    diff = max_len - len(result_dict[key])
                    result_dict[key] = result_dict[key] + [[]]*diff
        df_result = pd.DataFrame(result_dict)
        df_result.set_index('index', inplace=True)
        df_result.sort_index(inplace=True)
        output_name = dataset_name.split("/")[-1].replace(".json","")+"_NER.json"
        df_result.to_json(output_name)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
