#%%
import yaml
import argparse
import pandas as pd
import os
import sys
import time
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

from data_preprocessing import preprocess_NCBI, preprocess_TESSy
from construct_vocab import construct_MM_vocab

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--preprocess_TESSy", action="store_true", help="Prepare TESSy data")
    argparser.add_argument("--preprocess_NCBI", action="store_true", help="Prepare NCBI data")
    argparser.add_argument("--preprocess_both", action="store_true", help="Prepare both TESSy and NCBI data")
    argparser.add_argument("--construct_vocab", action="store_true", help="Construct vocabulary")
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config_MM.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    args = argparser.parse_args()
    if not any([args.preprocess_TESSy, args.preprocess_NCBI, args.preprocess_both, args.construct_vocab]):
        print("No action specified. Will prepare both TESSy and NCBI data and construct vocabulary.")
        args.preprocess_both = True
        args.construct_vocab = True
    if args.preprocess_both:
        print("Preparing both TESSy and NCBI data...\n")
        args.preprocess_TESSy = True
        args.preprocess_NCBI = True
    data_dict = config['data']
    if args.preprocess_NCBI:
        NCBI_start = time.time()
        print("Preprocessing NCBI data...")
        ds_NCBI = preprocess_NCBI(
            path=data_dict['NCBI']['raw_path'],
            save_path=data_dict['NCBI']['save_path'],
            include_phenotype=data_dict['NCBI']['include_phenotype'],
            ab_names_to_abbr=data_dict['antibiotics']['name_to_abbr'],
            exclude_antibiotics=data_dict['exclude_antibiotics'], 
            threshold_year=data_dict['NCBI']['threshold_year'],
            exclude_genotypes=data_dict['NCBI']['exclude_genotypes'],
            exclude_assembly_variants=data_dict['NCBI']['exclude_assembly_variants'],
            exclusion_chars=data_dict['NCBI']['exclusion_chars'],
            gene_count_threshold=data_dict['NCBI']['gene_count_threshold']
        )
        NCBI_time = time.time() - NCBI_start
        if NCBI_time > 60:
            disp_time = f"{NCBI_time/60:.2f} minutes"
        else:
            disp_time = f"{NCBI_time:.2f} seconds"
        print(f"Preprocessing NCBI completed in {disp_time}.")
        print()
    if args.preprocess_TESSy:
        TESSy_start = time.time()
        print("Preprocessing TESSy data...")
        ds_TESSy = preprocess_TESSy(
            path=data_dict['TESSy']['raw_path'],
            pathogens=data_dict['pathogens'],
            save_path=data_dict['TESSy']['save_path'],
            exclude_antibiotics=data_dict['exclude_antibiotics'],
            impute_age=data_dict['TESSy']['impute_age'],
            impute_gender=data_dict['TESSy']['impute_gender']
        )
        TESSy_time = time.time() - TESSy_start
        if TESSy_time > 60:
            disp_time = f"{TESSy_time/60:.2f} minutes"
        else:
            disp_time = f"{TESSy_time:.2f} seconds"
        print(f"Preprocessing TESSy completed in {disp_time}.")
    if any([args.preprocess_TESSy, args.preprocess_NCBI]):
        print("Preprocessing complete.\n")
        
    if args.construct_vocab:
        print("Constructing vocabulary...")
        if data_dict['exclude_antibiotics']:
            antibiotics = sorted(list(set(data_dict['antibiotics']['abbr_to_name'].keys()) - set(data_dict['exclude_antibiotics'])))
        else:
            antibiotics = sorted(list(data_dict['antibiotics']['abbr_to_name'].keys()))
        print(f"{len(antibiotics)} antibiotics: {antibiotics}")
        specials = config['specials']
        print("Loading preprocessed datasets...")
        print(f"NCBI: {data_dict['NCBI']['save_path']}")
        df_geno = pd.read_pickle(data_dict['NCBI']['save_path'])
        print(f"TESSy: {data_dict['TESSy']['save_path']}")
        df_pheno = pd.read_pickle(data_dict['TESSy']['save_path'])
        vocab = construct_MM_vocab(
            df_geno,
            df_pheno,
            antibiotics,
            specials,
            savepath_vocab=config['savepath_vocab'],
        )
        print("Vocabulary constructed.")
        print(f"Vocab size: {len(vocab)}")