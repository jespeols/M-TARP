import torch
import yaml
import argparse
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

# user-defined modules
from model import Model
from datasets.PretrainDataset import PretrainDataset
from trainers.Pretrainer import Pretrainer

# user-defined functions
from construct_vocab import construct_vocab
from utils import get_multimodal_split_indices, export_results
from data.NCBI_parser import parse_NCBI
from data.TESSy_parser import parse_TESSy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args(config):
    """Parse command-line arguments and overwrite config values."""
    parser = argparse.ArgumentParser()
    
    config_pt = config['pretraining']
    
    # Basic arguments
    parser.add_argument("--wandb_mode", type=str, default=config_pt.get('wandb_mode'))
    parser.add_argument("--name", type=str, default=config_pt.get('name'))
    parser.add_argument("--exp_folder", type=str, help="Folder to save experiment results", default=config_pt.get('exp_folder'))
    
    # Model parameters
    parser.add_argument("--num_layers", type=int, default=config_pt.get('num_layers'))
    parser.add_argument("--num_heads", type=int, default=config_pt.get('num_heads'))
    parser.add_argument("--emb_dim", type=int, default=config_pt.get('emb_dim'))
    parser.add_argument("--ff_dim", type=int, default=config_pt.get('ff_dim'))
    parser.add_argument("--hidden_dim", type=int, default=config_pt.get('hidden_dim'))
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=config_pt.get('batch_size'))
    parser.add_argument("--epochs", type=int, default=config_pt.get('epochs'))
    parser.add_argument("--lr", type=float, default=config_pt.get('lr'))
    parser.add_argument("--wl_strength", type=str, choices=['mild', 'strong'], help="Weighted loss strength", default=config_pt.get('wl_strength'))
    parser.add_argument("--random_state", type=int, default=config_pt.get('random_state'))
    
    # Masking and data preparation
    parser.add_argument("--mask_prob_geno", type=float, default=config_pt.get('mask_prob_geno'))
    parser.add_argument("--mask_prob_pheno", type=float, default=config_pt.get('mask_prob_pheno'))
    parser.add_argument("--num_known_ab", type=int, default=config_pt.get('num_known_ab'))
    parser.add_argument("--num_known_classes", type=int, default=config_pt.get('num_known_classes'))
    parser.add_argument("--always_mask_replace", action="store_true", help="Always replace masked tokens", default=config_pt.get('always_mask_replace'))
    
    # Data preparation
    parser.add_argument("--parse_TESSy", action="store_true", help="Prepare TESSy data", default=config['data']['TESSy'].get('parse_data'))
    parser.add_argument("--parse_NCBI", action="store_true", help="Prepare NCBI data", default=config['data']['NCBI'].get('parse_data'))
    parser.add_argument("--no_eval", action="store_true", help="Disable evaluation", default=not config_pt.get('do_eval'))

    return parser.parse_args()


def update_config_with_args(config, args):
    """Overwrite config dictionary with provided command-line arguments."""
    
    config_pt = config['pretraining']
    # Masking method assertion
    assert sum([
        args.mask_prob_pheno is not None,
        args.num_known_ab is not None,
        args.num_known_classes is not None
    ]) <= 1, "Choose only one masking method."
    
    # Set masking method
    if args.mask_prob_pheno:
        config_pt.update({
            'masking_method': 'random',
            'mask_prob_pheno': args.mask_prob_pheno,
            'num_known_ab': None,
            'num_known_classes': None
        })
    elif args.num_known_ab:
        config_pt.update({
            'masking_method': 'num_known_ab',
            'num_known_ab': args.num_known_ab,
            'mask_prob_pheno': None,
            'num_known_classes': None
        })
    elif args.num_known_classes:
        config_pt.update({
            'masking_method': 'num_known_classes',
            'num_known_classes': args.num_known_classes,
            'mask_prob_pheno': None,
            'num_known_ab': None
        })

    # Update rest of the config values
    config_pt.update({
        'wandb_mode': args.wandb_mode,
        'name': args.name,
        'exp_folder': args.exp_folder,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'emb_dim': args.emb_dim,
        'ff_dim': args.ff_dim,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'wl_strength': args.wl_strength,
        'always_mask_replace': args.always_mask_replace,
        'random_state': args.random_state,
        'do_eval': not args.no_eval
    })
    config['data']['TESSy'].update({'parse_data': args.parse_TESSy})
    config['data']['NCBI'].update({'parse_data': args.parse_NCBI})


if __name__ == "__main__":    
    print(f"\nCurrent working directory: {os.getcwd()}")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")  
    print("Loading config file...")
    
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    config_pt = config['pretraining']
    # overwrite config with command line arguments
    args = parse_args(config)
    update_config_with_args(config, args)
        
    os.environ['WANDB_MODE'] = config_pt['wandb_mode']
    if config_pt['exp_folder']:
        p = Path(BASE_DIR / "results" / config_pt['exp_folder'])
    else:
        p = Path(BASE_DIR / "results")
    if config_pt['name']:
        results_dir = Path(os.path.join(p, config_pt['name']))
    else:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(os.path.join(BASE_DIR / "results", "experiment_" + str(time_str)))
    print(f"Name of experiment: {config_pt['name']}")
    print(f"Results directory: {results_dir}")
    
    print("\nLoading dataset...")
    data_dict = config['data']
    ########################## Genotype dataset ##########################
    if data_dict['NCBI']['parse_data']:
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
    else:
        print(f"Loading preprocessed NCBI data from {data_dict['NCBI']['load_path']}...")
        ds_NCBI = pd.read_pickle(os.path.join(BASE_DIR, data_dict['NCBI']['load_path']))
         
    specials = config['specials']
    pad_token = specials['PAD']
    pad_idx = list(specials.values()).index(pad_token) # pass to model for embedding
    ds_geno = ds_NCBI[ds_NCBI['num_ab'] == 0].reset_index(drop=True)
    ds_geno.fillna(pad_token, inplace=True)
    
    ############################# Phenotype dataset ##########################
    if data_dict['TESSy']['parse_data']:
        ds_TESSy = parse_TESSy(
            path=data_dict['TESSy']['raw_path'],
            pathogens=data_dict['pathogens'],
            save_path=data_dict['TESSy']['save_path'],
            exclude_antibiotics=data_dict['exclude_antibiotics'],
        )
    else:
        print(f"Loading preprocessed TESSy data from {data_dict['TESSy']['load_path']}...")
        ds_TESSy = pd.read_pickle(os.path.join(BASE_DIR, data_dict['TESSy']['load_path']))
    ds_pheno = ds_TESSy.copy()
        
    abbr_to_class_enc = data_dict['antibiotics']['abbr_to_class_enc']
    ds_pheno['ab_classes'] = ds_pheno['phenotypes'].apply(lambda x: [abbr_to_class_enc[p.split('_')[0]] for p in x])
        
    ## construct vocabulary
    print(f"Constructing vocabulary using instructions from {config_path} ...")
    antibiotics = sorted(list(set(data_dict['antibiotics']['abbr_to_name'].keys()) - set(data_dict['exclude_antibiotics'])))
    vocab = construct_vocab(
        df_geno=ds_NCBI,
        df_pheno=ds_pheno,
        antibiotics=antibiotics,
        specials=specials,
        savepath_vocab=config['savepath_vocab']
    )
    vocab_size = len(vocab)
        
    if config['max_seq_len'] == 'auto':
        max_seq_len = int((ds_NCBI['num_genotypes'] + ds_NCBI['num_ab']).max() + 3)
    else:
        max_seq_len = config['max_seq_len']
        
    train_indices, val_indices = get_multimodal_split_indices(
        [ds_geno.shape[0], ds_pheno.shape[0]], 
        val_share=config_pt['val_share'], 
        random_state=config_pt['random_state']
    )
    ds_pt_train = PretrainDataset(
        ds_geno=ds_geno.iloc[train_indices[0]],
        ds_pheno=ds_pheno.iloc[train_indices[1]],
        vocab=vocab,
        antibiotics=antibiotics,
        specials=specials,
        max_seq_len=max_seq_len,
        always_mask_replace=config_pt['always_mask_replace'],
        mask_prob_geno=config_pt['mask_prob_geno'],
        masking_method=config_pt['masking_method'],
        mask_prob_pheno=config_pt['mask_prob_pheno'],
        num_known_ab=config_pt['num_known_ab'],
        num_known_classes=config_pt['num_known_classes'],
        random_state=config_pt['random_state']
    )
    ds_pt_val = PretrainDataset(
        ds_geno=ds_geno.iloc[val_indices[0]],
        ds_pheno=ds_pheno.iloc[val_indices[1]],
        vocab=vocab,
        antibiotics=antibiotics,
        specials=specials,
        max_seq_len=max_seq_len,
        always_mask_replace=config_pt['always_mask_replace'],
        mask_prob_geno=config_pt['mask_prob_geno'],
        masking_method=config_pt['masking_method'],
        mask_prob_pheno=config_pt['mask_prob_pheno'],
        num_known_ab=config_pt['num_known_ab'],
        num_known_classes=config_pt['num_known_classes'],
        random_state=config_pt['random_state']
    )
    model = Model(
        config, 
        vocab_size, 
        max_seq_len, 
        len(antibiotics), 
        pad_idx=pad_idx, 
        random_state=config_pt['random_state']
    ).to(device)
    trainer = Pretrainer(
        config=config,
        model=model,
        antibiotics=antibiotics,
        train_set=ds_pt_train,
        val_set=ds_pt_val,
        results_dir=results_dir,
    )
    trainer.print_model_summary()
    trainer.print_trainer_summary()
    results = trainer()    
    print("Finished training!")
    print("Exporting results...") 
    export_results(results, results_dir / "pt_results.pkl")