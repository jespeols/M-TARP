import torch
import yaml
import wandb
import argparse
import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.model_selection import KFold
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

# user-defined modules
from model import Model
from datasets.FinetuneDataset import FinetuneDataset
from trainers.Finetuner import Finetuner

# user-defined functions
from utils import get_split_indices, export_results, get_average_and_std_df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_wandb(project_name: str, wandb_name: str, config: dict):
    wandb_run = wandb.init(project=project_name, name=wandb_name, config=config)
    wandb_run.define_metric("fold", hidden=True)
    
    wandb_run.define_metric("Losses/train_loss", summary="min", step_metric="fold")
    wandb_run.define_metric("Losses/val_loss", summary="min", step_metric="fold")
    wandb_run.define_metric("Accuracies/val_acc", summary="max", step_metric="fold")
    wandb_run.define_metric("Accuracies/val_iso_acc", summary="max", step_metric="fold")
    wandb_run.define_metric("Class_metrics/val_sens", summary="max", step_metric="fold")
    wandb_run.define_metric("Class_metrics/val_spec", summary="max", step_metric="fold")
    wandb_run.define_metric("Class_metrics/val_F1", summary="max", step_metric="fold")
    wandb_run.define_metric("Class_metrics/val_auc_score", summary="max", step_metric="fold")
    
    wandb_run.define_metric("Losses/avg_val_loss")
    wandb_run.define_metric("Accuracies/avg_val_acc")
    wandb_run.define_metric("Accuracies/avg_val_iso_acc")
    wandb_run.define_metric("Class_metrics/avg_val_sens")
    wandb_run.define_metric("Class_metrics/avg_val_spec")
    wandb_run.define_metric("Class_metrics/avg_val_F1")
    wandb_run.define_metric("Class_metrics/avg_val_auc_score")

    return wandb_run


def list_of_floats(arg):
    try:
        return list(map(float, arg.split(',')))
    except:
        raise argparse.ArgumentTypeError("Argument must be a list of floats separated by commas")


def list_of_strings(arg):
    try:
        return list(map(str, arg.split(',')))
    except:
        raise argparse.ArgumentTypeError("Argument must be a list of strings separated by commas")


def parse_args(config_ft):
    """Parse command-line arguments and return an updated config."""
    parser = argparse.ArgumentParser()

    # Experiment setup
    parser.add_argument("--wandb_mode", type=str, default=config_ft.get('wandb_mode'), help="Wandb mode")
    parser.add_argument("--name", type=str, default=config_ft.get('name'), help="Name of experiment")
    parser.add_argument("--exp_folder", type=str, help="Name of experiment folder", default=config_ft.get('exp_folder'))
    parser.add_argument("--model_path", type=str, help="Path to model state dict")
    parser.add_argument("--ds_path", type=str, default=config_ft.get('ds_path'), help="Path to dataset")
    parser.add_argument("--no_pt", action="store_true", help="Use randomly initialized model")
    
    # Model masking and filtering
    parser.add_argument("--mask_prob_geno", type=float, default=config_ft.get('mask_prob_geno'), help="Masking probability for genotypes")
    parser.add_argument("--no_geno_masking", action="store_true", help="Disable geno masking", default=config_ft.get('no_geno_masking'))
    parser.add_argument("--mask_prob_pheno", type=float, default=config_ft.get('mask_prob_pheno'), help="Masking probability for phenotypes")
    parser.add_argument("--num_known_ab", type=int, default=config_ft.get('num_known_ab'), help="Number of known antibiotics")
    parser.add_argument("--num_known_classes", type=int, default=config_ft.get('num_known_classes'), help="Number of known classes")
    parser.add_argument("--min_num_ab", type=int, help="Limit dataset to isolates with a minimum number of antibiotics")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=config_ft.get('batch_size'), help="Batch size")
    parser.add_argument("--epochs", type=int, default=config_ft.get('epochs'), help="Number of epochs")
    parser.add_argument("--lr", type=float, default=config_ft.get('lr'), help="Learning rate")
    parser.add_argument("--random_state", type=int, default=config_ft.get('random_state'), help="Random state")
    
    # Loss function
    parser.add_argument("--wl_strength", type=str, choices=['mild', 'strong'], help="Weighted loss strength", default=config_ft.get('wl_strength'))

    # Cross-validation and model saving
    parser.add_argument("--train_shares", type=list, default=config_ft.get('train_shares'), help="List of shares for training sizes")
    parser.add_argument("--no_cv", action="store_true", help="Disable cross-validation")
    parser.add_argument("--val_share", type=float, default=config_ft.get('val_share'), help="Validation share when CV is disabled")
    parser.add_argument("--num_folds", type=int, default=config_ft.get('num_folds'), help="Number of folds for cross-validation")
    parser.add_argument("--save_model", action="store_true", help="Save model (last fold if CV is enabled)", default=config_ft.get('save_model'))
    parser.add_argument("--save_best_model", action="store_true", help="Save best-performing model")
    
    return parser.parse_args()


def update_config_with_args(config_ft, args):
    """Update the config based on the parsed arguments."""
    
    if not args.no_pt:
        assert args.model_path, "Path to model state dict must be provided using pretrained model."
    
    config_ft.update({
        'wandb_mode': args.wandb_mode,
        'name': args.name,
        'exp_folder': args.exp_folder,
        'model_path': None if args.no_pt else args.model_path,
        'ds_path': args.ds_path,
        'no_pt': args.no_pt,
        'mask_prob_geno': args.mask_prob_geno,
        'no_geno_masking': args.no_geno_masking,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'random_state': args.random_state,
        'wl_strength': args.wl_strength,
        'save_model': args.save_model,
    })
    
    # Handle masking methods (only one can be chosen)
    assert sum([
        args.mask_prob_pheno is not None,
        args.num_known_ab is not None,
        args.num_known_classes is not None
    ]) <= 1, "Choose only one masking method."

    if args.mask_prob_pheno:
        config_ft.update({'masking_method': 'random', 'mask_prob_pheno': args.mask_prob_pheno})
        config_ft['num_known_ab'], config_ft['num_known_classes'] = None, None
    elif args.num_known_ab:
        config_ft.update({'masking_method': 'num_known_ab', 'num_known_ab': args.num_known_ab})
        config_ft['mask_prob_pheno'], config_ft['num_known_classes'] = None, None
    elif args.num_known_classes:
        config_ft.update({'masking_method': 'num_known_classes', 'num_known_classes': args.num_known_classes})
        config_ft['mask_prob_pheno'], config_ft['num_known_ab'] = None, None
    
    # Handle cross-validation
    if not args.no_cv:
        config_ft['num_folds'] = args.num_folds if args.num_folds else config_ft['num_folds']
        config_ft['val_share'] = 1 / config_ft['num_folds']
    else:
        config_ft.update({'num_folds': None, 'val_share': args.val_share if args.val_share else config_ft['val_share']})


if __name__ == "__main__":    
        
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")  
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config_ft = config['finetuning']
    data_dict = config['data']
    
    # overwrite config with command line arguments
    args = parse_args(config_ft)
    update_config_with_args(config_ft, args)
    train_shares = args.train_shares if args.train_shares else [0.8]
    
    os.environ['WANDB_MODE'] = config_ft['wandb_mode']
    
    print(f"\nLoading dataset from {os.path.join(BASE_DIR, config_ft['ds_path'])}...")
    ds_NCBI = pd.read_pickle(BASE_DIR / data_dict['NCBI']['load_path'])
    ds_MM = ds_NCBI[ds_NCBI['num_ab'] > 0].sample(frac=1, random_state=config_ft['random_state']).reset_index(drop=True)
    abbr_to_class_enc = data_dict['antibiotics']['abbr_to_class_enc']
    ds_MM['ab_classes'] = ds_MM['phenotypes'].apply(lambda x: [abbr_to_class_enc[p.split('_')[0]] for p in x])
    if args.min_num_ab:
        print(f"Filtering out isolates with less than {args.min_num_ab} antibiotics...")
        ds_MM = ds_MM[ds_MM['num_ab'] >= args.min_num_ab].reset_index(drop=True)
   
    print("Loading vocabulary...")
    vocab = torch.load(BASE_DIR / config_ft['loadpath_vocab'])
    specials = config['specials']
    pad_token = specials['PAD']
    ds_MM.fillna(pad_token, inplace=True)

    antibiotics = sorted(list(set(data_dict['antibiotics']['abbr_to_name'].keys()) - set(data_dict['exclude_antibiotics'])))
    vocab_size = len(vocab)
    if config['max_seq_len'] == 'auto':
        max_seq_len = int((ds_NCBI['num_genotypes'] + ds_NCBI['num_ab']).max() + 3)
    else:
        max_seq_len = config['max_seq_len']
    
    run_name = config_ft['name']
    print(f"Name of run: {run_name}")
    for i, train_share in enumerate(train_shares):
        print(f"Train share {i+1} of {len(train_shares)}: {train_share:.0%}")
        if not train_share == 0.8:
            config_ft['name'] = f"{run_name}_train_share{train_share}"
        else:
            config_ft['name'] = run_name
        if args.exp_folder:
            config_ft['exp_folder'] = args.exp_folder
            p = Path(BASE_DIR / "results" / args.exp_folder)
        else:
            p = Path(BASE_DIR / "results")
        results_dir = Path(os.path.join(p, config_ft['name'])) 
        
        train_losses = []
        losses = []
        accs = []
        iso_accs = []
        sensitivities = []
        specificities = []
        F1_scores = []
        auc_scores = []
        roc_results = []
        iso_stats = []
        ab_stats = []
        
        num_folds = config_ft['num_folds']
        if num_folds:
            print(f"Splitting dataset into {num_folds} folds...")
            kf = KFold(n_splits=num_folds)
            indices_list = kf.split(ds_MM.index)
        else:
            print("No cross-validation")
            train_indices, val_indices = get_split_indices(
                ds_MM.shape[0], 
                val_share=config_ft['val_share'], 
                random_state=config_ft['random_state']
            )
            indices_list = [(train_indices, val_indices)]
            
        best_fold, best_val_loss = 0, float('inf')
        for j, (train_indices, val_indices) in enumerate(indices_list):
            print("="*80)
            print("="*80)
            if num_folds:
                print(f"Training fold {j+1} of {num_folds}...")
                print("="*80)

            # adjust train size depending on train_share (intended as share of TOTAL dataset, not train set of fold)
            if train_share < 0.8:
                train_size = int(len(train_indices) * train_share/(1-config_ft['val_share']))
                train_indices = train_indices[:train_size]
            
            ds_ft_train = FinetuneDataset(
                df_MM=ds_MM.iloc[train_indices],
                vocab=vocab,
                antibiotics=antibiotics,
                specials=specials,
                max_seq_len=max_seq_len,
                masking_method=config_ft['masking_method'],
                mask_prob_geno=config_ft['mask_prob_geno'],
                mask_prob_pheno=config_ft['mask_prob_pheno'],
                num_known_ab=config_ft['num_known_ab'],
                num_known_classes=config_ft['num_known_classes'],
                always_mask_replace=config_ft['always_mask_replace'],
                random_state=config_ft['random_state'],
                no_geno_masking=config_ft['no_geno_masking']
            )
            ds_ft_val = FinetuneDataset(
                df_MM=ds_MM.iloc[val_indices],
                vocab=vocab,
                antibiotics=antibiotics,
                specials=specials,
                max_seq_len=max_seq_len,
                masking_method=config_ft['masking_method'],
                mask_prob_geno=config_ft['mask_prob_geno'],
                mask_prob_pheno=config_ft['mask_prob_pheno'],
                num_known_ab=config_ft['num_known_ab'],
                num_known_classes=config_ft['num_known_classes'],
                always_mask_replace=config_ft['always_mask_replace'],
                random_state=config_ft['random_state'],
                no_geno_masking=config_ft['no_geno_masking']
            )
            pad_idx = vocab[pad_token]
            model = Model(
                config, 
                vocab_size, 
                max_seq_len, 
                len(antibiotics), 
                pad_idx, 
                pheno_only=True,
                random_state=config_ft['random_state']
            ).to(device)
            tuner = Finetuner(
                config=config,
                model=model,
                antibiotics=antibiotics,
                train_set=ds_ft_train,
                val_set=ds_ft_val,
                results_dir=results_dir,
                ds_size=ds_MM.shape[0],
                CV_mode=True if num_folds else False,
            )
            if not config_ft['no_pt']:
                tuner.load_model(Path(BASE_DIR / config_ft['model_path']))
                tuner.model.is_pretrained = True
            if j == 0:
                tuner.print_model_summary()
                tuner.print_trainer_summary()
                wandb_config = {
                    "trainer_type": "fine-tuning",
                    "exp_folder": config_ft['exp_folder'],
                    "Device:" : f"{device.type} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else device.type,
                    "epochs": tuner.epochs,
                    "batch_size": tuner.batch_size,
                    "hidden_dim": tuner.model.hidden_dim,
                    "num_layers": tuner.model.num_layers,
                    "num_heads": tuner.model.num_heads,
                    "emb_dim": tuner.model.emb_dim,
                    'ff_dim': tuner.model.ff_dim,
                    "lr": tuner.lr,
                    "ab_weights": tuner.ab_weights if tuner.wl_strength else None,
                    "weight_decay": tuner.weight_decay,
                    "masking_method": tuner.masking_method, 
                    "mask_prob_geno": tuner.mask_prob_geno,
                    "mask_prob_pheno": tuner.mask_prob_pheno,
                    "num_known_ab": tuner.num_known_ab,
                    "num_known_classes": tuner.num_known_classes,
                    "max_seq_len": tuner.model.max_seq_len,
                    "vocab_size": len(vocab),
                    "num_parameters": sum(p.numel() for p in tuner.model.parameters() if p.requires_grad),
                    "num_antibiotics": tuner.num_ab,
                    "antibiotics": tuner.antibiotics,
                    "train_share": train_share,
                    "train_size": tuner.train_size,
                    "CV_mode": tuner.CV_mode,
                    'val_share': round(tuner.val_share, 2),
                    "val_size": tuner.val_size,
                    "is_pretrained": tuner.model.is_pretrained,
                }
                wandb_run = init_wandb(config_ft['project_name'], config_ft['name'], wandb_config)
                
            ft_results = tuner()
            log_dict = {
                "fold": j+1,
                "Losses/train_loss": ft_results['train_loss'],
                "Losses/val_loss": ft_results['loss'],
                "Accuracies/val_acc": ft_results['acc'],
                "Accuracies/val_iso_acc": ft_results['iso_acc'],
                "Class_metrics/val_sens": ft_results['sens'],
                "Class_metrics/val_spec": ft_results['spec'],
                "Class_metrics/val_F1": ft_results['F1'],
                "Class_metrics/val_auc_score": ft_results['auc_score'],
            }
            wandb_run.log(log_dict)
            
            train_losses.append(ft_results['train_loss'])
            losses.append(ft_results['loss'])
            accs.append(ft_results['acc'])
            iso_accs.append(ft_results['iso_acc'])
            sensitivities.append(ft_results['sens'])
            specificities.append(ft_results['spec'])
            F1_scores.append(ft_results['F1'])
            auc_scores.append(ft_results['auc_score'])
            roc_results.append(ft_results['roc'])
            iso_stats.append(ft_results['iso_stats'])
            ab_stats.append(ft_results['ab_stats'])
            
            if num_folds:
                if ft_results['loss'] < best_val_loss:
                    best_val_loss = ft_results['loss']
                    best_fold = j
                    if args.save_best_model:
                        best_model_state = tuner.model.state_dict()
        if num_folds:
            print("All folds completed!")
        if args.save_best_model:
            print(f"Saving model from best-performing fold ({best_fold+1})...")
            torch.save(best_model_state, results_dir / 'best_model_state.pt')
            print(f"Model saved to {results_dir / 'best_model_state.pt'}")
            
        CV_results = {
            'train_losses': train_losses,
            'losses': losses,
            'accs': accs,
            'iso_accs': iso_accs,
            'sensitivities': sensitivities,
            'specificities': specificities,
            'F1_scores': F1_scores,
            "auc_scores": auc_scores,
            "roc_results": roc_results,
            'iso_stats': iso_stats,
            'ab_stats': ab_stats
        }  
        df_CV = get_average_and_std_df(CV_results, include_auc=True)
        log_dict = {
            "Losses/avg_val_loss": df_CV.loc['loss', 'avg'],
            "Accuracies/avg_val_acc": df_CV.loc['accuracy', 'avg'],
            "Accuracies/avg_val_iso_acc": df_CV.loc["isolate accuracy", 'avg'],
            "Class_metrics/avg_val_sens": df_CV.loc["sensitivity", 'avg'],
            "Class_metrics/avg_val_spec": df_CV.loc["specificity", 'avg'],
            "Class_metrics/avg_val_F1": df_CV.loc["F1", 'avg'],
            "Class_metrics/avg_val_auc_score": df_CV.loc["auc_score", 'avg'],
        }
        wandb_run.log(log_dict)
        wandb_run.finish()
        print("Exporting results...")    
        export_results(CV_results, results_dir / 'CV_results.pkl')