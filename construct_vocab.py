import os
import torch
import pandas as pd
import yaml
import torchtext; torchtext.disable_torchtext_deprecation_warning()

from pathlib import Path
from torchtext.vocab import vocab as Vocab
from itertools import chain
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent

def construct_vocab(
        df_geno: pd.DataFrame,
        df_pheno: pd.DataFrame,
        antibiotics: list,
        specials: dict,
        savepath_vocab: Path = None
    ):
    token_counter = Counter()
    ds_geno = df_geno.copy()
    ds_pheno = df_pheno.copy()
    
    PAD, UNK = specials['PAD'], specials['UNK']
    special_tokens = specials.values()
    
    year_geno = ds_geno[ds_geno['year'] != PAD]['year'].astype('Int16')
    min_year = min(year_geno.min(), ds_pheno['year'].min())
    max_year = max(year_geno.max(), ds_pheno['year'].max())
    year_range = range(min_year, max_year+1)
    token_counter.update([str(y) for y in year_range])
    
    min_age, max_age = ds_pheno['age'].min(), ds_pheno['age'].max()
    age_range = range(int(min_age), int(max_age+1))
    token_counter.update([str(a) for a in age_range])
    
    genders = ds_pheno['gender'].unique().astype(str).tolist()
    token_counter.update(genders)
    
    pheno_countries = ds_pheno['country'].sort_values().unique()
    geno_countries = ds_geno['country'].sort_values().dropna().unique()
    countries = set(pheno_countries).union(set(geno_countries))
    token_counter.update(countries)
    
    token_counter.update(list(chain(*ds_geno['genotypes'])))
    token_counter.update([ab + '_' + res for ab in antibiotics for res in ['R', 'S']])  
    
    vocab = Vocab(token_counter, specials=special_tokens)
    vocab.set_default_index(vocab[UNK])
    if savepath_vocab:
        print(f"Saving vocabulary to {savepath_vocab}")
        torch.save(vocab, savepath_vocab)
    return vocab

if __name__ == '__main__':   
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
        
    data_dict = config['data']
    print(f"Constructing vocabulary using instructions from {config_path} ...")
    if data_dict['exclude_antibiotics']:
        antibiotics = sorted(list(set(data_dict['antibiotics']['abbr_to_name'].keys()) - set(data_dict['exclude_antibiotics'])))
    else:
        antibiotics = sorted(list(data_dict['antibiotics']['abbr_to_name'].keys()))
    print(f"{len(antibiotics)} antibiotics: {antibiotics}")
    specials = config['specials']
    print("Loading parsed datasets...")
    print(f"NCBI: {data_dict['NCBI']['save_path']}")
    df_geno = pd.read_pickle(data_dict['NCBI']['save_path'])
    print(f"TESSy: {data_dict['TESSy']['save_path']}")
    df_pheno = pd.read_pickle(data_dict['TESSy']['save_path'])
    vocab = construct_vocab(
        df_geno,
        df_pheno,
        antibiotics,
        specials,
        savepath_vocab=config['savepath_vocab'],
    )
    print("Vocabulary constructed.")
    print(f"Vocab size: {len(vocab)}")