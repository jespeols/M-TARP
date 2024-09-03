import torch
import yaml
import wandb
import argparse
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

# user-defined modules
from multimodal.models import BERT
from multimodal.datasets import MMFinetuneDataset
from multimodal.trainers import MMBertFineTuner

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


if __name__ == "__main__":    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wandb_mode", type=str)
    argparser.add_argument("--name", type=str)
    argparser.add_argument("--exp_folder", type=str, help="Name of experiment folder")
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--ds_path", type=str)
    argparser.add_argument("--no_pt", action="store_true", help="Enable naive model")
    argparser.add_argument("--mask_prob_geno", type=float)
    argparser.add_argument("--no_geno_masking", action="store_true", help="Disable geno masking")
    argparser.add_argument("--mask_prob_pheno", type=float)
    argparser.add_argument("--num_known_ab", type=int)
    argparser.add_argument("--num_known_classes", type=int)
    argparser.add_argument("--filter_genes_by_ab_class", type=list_of_strings, help="Filter genes by antibiotic classes provided in list")
    argparser.add_argument("--filter_isolates_by_ab_class", type=list_of_strings, help="Filter isolates by presence of genes associated with antibiotic classes provided in list")
    argparser.add_argument("--min_num_ab", type=int)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--loss_fn", type=str, help="Loss function to use")
    argparser.add_argument("--wl_strength", type=str, help="Strength of weighted CE loss functions for antibiotics ('mild' or 'strong')")
    argparser.add_argument("--gamma", type=float, help="Gamma parameter for focal loss")
    argparser.add_argument("--lr", type=float)
    argparser.add_argument("--random_state", type=int)
    argparser.add_argument("--train_shares", type=list_of_floats, help="List of shares for training sizes to indices_list over")
    argparser.add_argument("--no_cv", action="store_true", help="Disable cross-validation")
    argparser.add_argument("--val_share", type=float, help="Validation share when CV is disabled")
    argparser.add_argument("--num_folds", type=int)
    argparser.add_argument("--save_model", action="store_true", help="Save model (from last fold if CV is enabled)")
    argparser.add_argument("--save_best_model", action="store_true", help="Save model from best-performing fold")
        
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")  
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config_MM.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config_ft = config['fine_tuning']
    data_dict = config['data']
    
    # overwrite config with command line arguments
    args = argparser.parse_args()
    config_ft['wandb_mode'] = args.wandb_mode if args.wandb_mode else config_ft['wandb_mode']
    config_ft['name'] = args.name if args.name else config_ft['name']
    if args.save_model:
        config_ft['save_model'] = True
    if args.model_path:
        config_ft['model_path'] = args.model_path
    elif not config_ft['model_path']:
        args.no_pt = True
    config_ft['ds_path'] = args.ds_path if args.ds_path else config_ft['ds_path']
    if args.no_pt:
        config_ft['model_path'] = None
        config_ft['no_pt'] = True
    config_ft['mask_prob_geno'] = args.mask_prob_geno if args.mask_prob_geno else config_ft['mask_prob_geno']
    config_ft['no_geno_masking'] = args.no_geno_masking if args.no_geno_masking else config_ft['no_geno_masking']
    assert sum([
        (args.mask_prob_pheno is not None), (args.num_known_ab is not None), (args.num_known_classes is not None)
    ]) <= 1, "Choose only one masking method."
    if args.mask_prob_pheno:
        config_ft['masking_method'] = 'random'
        config_ft['mask_prob_pheno'] = args.mask_prob_pheno
        config_ft['num_known_ab'], config_ft['num_known_classes'] = None, None
    elif args.num_known_ab:
        config_ft['masking_method'] = 'num_known_ab'
        config_ft['num_known_ab'] = args.num_known_ab
        config_ft['mask_prob_pheno'], config_ft['num_known_classes'] = None, None
    elif args.num_known_classes:
        config_ft['masking_method'] = 'num_known_classes'
        config_ft['num_known_classes'] = args.num_known_classes
        config_ft['mask_prob_pheno'], config_ft['num_known_ab'] = None, None
    config_ft['batch_size'] = args.batch_size if args.batch_size else config_ft['batch_size']
    config_ft['epochs'] = args.epochs if args.epochs else config_ft['epochs']
    config_ft['random_state'] = args.random_state if args.random_state else config_ft['random_state']
    if args.loss_fn:
        if not args.loss_fn in ['focal', 'bce']:
            raise NotImplementedError("Invalid loss function, choose from ['focal', 'bce']")
        config_ft['loss_fn'] = args.loss_fn
    if args.wl_strength:
        assert args.wl_strength in ['mild', 'strong'], "Invalid weighted loss strength, choose from ['mild', 'strong']"
        config_ft['wl_strength'] = args.wl_strength   
    if args.gamma:
        assert config_ft['loss_fn'] == 'focal', 'Alpha and gamma parameters only available for focal loss function. Use weighted loss strength for BCE.'
        config_ft['gamma'] = args.gamma if args.gamma else config_ft['gamma']
    config_ft['lr'] = args.lr if args.lr else config_ft['lr']
    train_shares = args.train_shares if args.train_shares else [0.8]
    if not args.no_cv:
        config_ft['num_folds'] = args.num_folds if args.num_folds else config_ft['num_folds']
        config_ft['val_share'] = 1/config_ft['num_folds']
    else:
        config_ft['num_folds'] = None
        config_ft['val_share'] = args.val_share if args.val_share else config_ft['val_share']
    
    os.environ['WANDB_MODE'] = config_ft['wandb_mode']
    
    print(f"\nLoading dataset from {os.path.join(BASE_DIR, config_ft['ds_path'])}...")
    ds_NCBI = pd.read_pickle(BASE_DIR / data_dict['NCBI']['load_path'])
    ds_MM = ds_NCBI[ds_NCBI['num_ab'] > 0].sample(frac=1, random_state=config_ft['random_state']).reset_index(drop=True)
    # ds_MM = ds_MM[ds_MM['country'] != 'USA'].reset_index(drop=True) # smaller, non-American dataset
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
    
    if args.filter_genes_by_ab_class:
        data_dict['NCBI']['filter_genes_by_ab_class'] = args.filter_genes_by_ab_class
    if args.filter_isolates_by_ab_class:
        data_dict['NCBI']['filter_isolates_by_ab_class'] = args.filter_isolates_by_ab_class

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
            p = Path(BASE_DIR / "results" / "MM" / args.exp_folder)
        else:
            p = Path(BASE_DIR / "results" / "MM")
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
            
            ds_ft_train = MMFinetuneDataset(
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
                filter_isolates_by_ab_class=data_dict['NCBI']['filter_isolates_by_ab_class'],
                filter_genes_by_ab_class=data_dict['NCBI']['filter_genes_by_ab_class'],
                random_state=config_ft['random_state'],
                no_geno_masking=config_ft['no_geno_masking']
            )
            ds_ft_val = MMFinetuneDataset(
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
            bert = BERT(config, vocab_size, max_seq_len, len(antibiotics), pad_idx, pheno_only=True).to(device)
            tuner = MMBertFineTuner(
                config=config,
                model=bert,
                antibiotics=antibiotics,
                train_set=ds_ft_train,
                val_set=ds_ft_val,
                results_dir=results_dir,
                ds_size=ds_MM.shape[0],
                CV_mode=True if num_folds else False,
            )
            if not config_ft['no_pt']:
                tuner.load_model(Path(BASE_DIR / 'results' / 'MM' / config_ft['model_path']))
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
                    "loss_fn": tuner.loss_fn,
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