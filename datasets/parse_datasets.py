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

from data.NCBI_parser import parse_NCBI
from data.TESSy_parser import parse_TESSy
from construct_vocab import construct_vocab

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--parse_TESSy", action="store_true", help="parse TESSy data")
    argparser.add_argument("--parse_NCBI", action="store_true", help="parse NCBI data")
    argparser.add_argument("--parse_both", action="store_true", help="parse both TESSy and NCBI data")
    argparser.add_argument("--construct_vocab", action="store_true", help="Construct vocabulary")
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    args = argparser.parse_args()
    if not any([args.parse_TESSy, args.parse_NCBI, args.parse_both, args.construct_vocab]):
        print("No action specified. Will parse both TESSy and NCBI data and construct vocabulary.")
        args.parse_both = True
        args.construct_vocab = True
    if args.parse_both:
        print("Parsing both TESSy and NCBI data...\n")
        args.parse_TESSy = True
        args.parse_NCBI = True
    data_dict = config['data']
    if args.parse_NCBI:
        NCBI_start = time.time()
        print("Parsing NCBI data...")
        ds_NCBI = parse_NCBI(
            path=data_dict['NCBI']['raw_path'],
            save_path=data_dict['NCBI']['save_path'],
            include_phenotype=data_dict['NCBI']['include_phenotype'],
            ab_names_to_abbr=data_dict['antibiotics']['name_to_abbr'],
            exclude_antibiotics=data_dict['exclude_antibiotics'], 
            threshold_year=data_dict['NCBI']['threshold_year'],
            exclude_assembly_variants=data_dict['NCBI']['exclude_assembly_variants'],
            gene_count_threshold=data_dict['NCBI']['gene_count_threshold']
        )
        NCBI_time = time.time() - NCBI_start
        if NCBI_time > 60:
            disp_time = f"{NCBI_time/60:.2f} minutes"
        else:
            disp_time = f"{NCBI_time:.1f} seconds"
        print(f"Parsing NCBI completed in {disp_time}.")
        print()
    if args.parse_TESSy:
        TESSy_start = time.time()
        print("Parsing TESSy data...")
        ds_TESSy = parse_TESSy(
            path=data_dict['TESSy']['raw_path'],
            pathogens=data_dict['pathogens'],
            save_path=data_dict['TESSy']['save_path'],
            exclude_antibiotics=data_dict['exclude_antibiotics'],
        )
        TESSy_time = time.time() - TESSy_start
        if TESSy_time > 60:
            disp_time = f"{TESSy_time/60:.2f} minutes"
        else:
            disp_time = f"{TESSy_time:.1f} seconds"
        print(f"Parsing TESSy completed in {disp_time}.")
    if any([args.parse_TESSy, args.parse_NCBI]):
        print("Parsing complete.\n")
        
    if args.construct_vocab:
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