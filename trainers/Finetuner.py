import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time 
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader
from pathlib import Path
from itertools import chain
from datetime import datetime
from sklearn.metrics import roc_curve, auc, roc_auc_score

from utils import WeightedBCEWithLogitsLoss, BinaryFocalWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MMBertFineTuner():
    
    def __init__(
        self,
        config: dict,
        model,
        antibiotics: list,
        train_set,
        val_set,
        results_dir: Path,
        ds_size: int = None,
        CV_mode: bool = False
    ):
        super(MMBertFineTuner, self).__init__()
        
        config_ft = config["fine_tuning"]
        self.random_state = config_ft['random_state']
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        
        self.model = model
        self.project_name = config_ft["project_name"]
        self.wandb_name = config_ft["name"] if config_ft["name"] else datetime.now().strftime("%Y%m%d-%H%M%S")
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics)
        
        self.train_set, self.train_size = train_set, len(train_set)
        self.val_set, self.val_size = val_set, len(val_set) 
        self.dataset_size = self.train_size + self.val_size
        if ds_size:
            self.dataset_size = ds_size
        else:
            self.dataset_size = self.train_size + self.val_size
        self.val_share, self.train_share = self.val_size / self.dataset_size, self.train_size / self.dataset_size
        self.batch_size = config_ft["batch_size"]
        self.val_batch_size = self.batch_size * 64
        self.num_batches = round(self.train_size / self.batch_size)
        self.vocab = self.train_set.vocab
        self.ab_to_idx = self.train_set.ab_to_idx
         
        self.lr = config_ft["lr"]
        self.weight_decay = config_ft["weight_decay"]
        self.epochs = config_ft["epochs"]
        self.patience = config_ft["early_stopping_patience"]
        self.save_model_ = config_ft["save_model"]
        
        self.masking_method = self.train_set.masking_method
        self.mask_prob_geno = self.train_set.mask_prob_geno
        self.mask_prob_pheno = self.train_set.mask_prob_pheno
        self.num_known_ab = self.train_set.num_known_ab
        self.num_known_classes = self.train_set.num_known_classes
        
        self.loss_fn = config_ft["loss_fn"]
        self.gamma = config_ft["gamma"]
        self.wl_strength = config_ft["wl_strength"] 
        if self.wl_strength:
            self.ab_weights = config['data']['antibiotics']['ab_weights_'+self.wl_strength]
            self.ab_weights = {ab: v for ab, v in self.ab_weights.items() if ab in self.antibiotics}
            self.alphas = [v for v in self.ab_weights.values()]
        else:   
            self.alphas = [0.5]*self.num_ab   ## equal class weights for all antibiotics
            
        if self.loss_fn == 'bce':
            self.ab_criterions = [WeightedBCEWithLogitsLoss(alpha=alpha).to(device) for alpha in self.alphas]
        elif self.loss_fn == 'focal':       ## TODO: Add individual parameter values for each antibiotic
            self.ab_criterions = [BinaryFocalWithLogitsLoss(alpha, self.gamma).to(device) for alpha in self.alphas]
        else:
            raise NotImplementedError("Only 'bce' and 'focal' functions are supported")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
                 
        self.current_epoch = 0
        self.CV_mode = CV_mode
        if self.CV_mode:
            self.num_folds = config_ft["num_folds"]
        self.report_every = config_ft["report_every"] 
        self.print_progress_every = config_ft["print_progress_every"]
        self._splitter_size = 80
        self.exp_folder = config_ft["exp_folder"]
        self.results_dir = results_dir
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
    def print_model_summary(self):        
        print("Model summary:")
        print("="*self._splitter_size)
        print(f"Is pre-trained: {'Yes' if self.model.is_pretrained else 'No'}")
        print(f"Embedding dim: {self.model.emb_dim}")
        print(f"Feed-forward dim: {self.model.ff_dim}")
        print(f"Hidden dim: {self.model.hidden_dim}")
        print(f"Number of heads: {self.model.num_heads}")
        print(f"Number of encoder layers: {self.model.num_layers}")
        print(f"Dropout probability: {self.model.dropout_prob:.0%}")
        print(f"Max sequence length: {self.model.max_seq_len}")
        print(f"Vocab size: {len(self.vocab):,}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("="*self._splitter_size)
        
    
    def print_trainer_summary(self):
        print("Trainer summary:")
        print("="*self._splitter_size)
        if device.type == "cuda":
            print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Device: {device}")        
        print(f"Training dataset size: {self.train_size:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {self.num_batches:,}")
        print(f"Number of antibiotics: {self.num_ab}")
        print(f"Antibiotics: {self.antibiotics}")
        if self.wl_strength:
            print("Antibiotic weights:", self.ab_weights)
        s = f"CV mode: {'On' if self.CV_mode else 'Off'}"
        if self.CV_mode:
            s += f" ({self.num_folds} folds)"
        print(s)
        print(f"Data split: {self.train_share:.0%} train | {self.val_share:.0%} val (size: {self.dataset_size:,})")
        if not self.train_set.no_genotype_masking:
            print(f"Mask probability for genotype: {self.train_set.mask_prob_geno:.0%}")
        else:
            print(f"No genotype masking")
        print(f"Masking method: {self.masking_method}")
        if self.masking_method == 'random':
            print(f"Mask probability for prediction task (phenotype): {self.mask_prob_pheno:.0%}")
        elif self.masking_method == 'num_known_ab':
            print(f"Number of known antibiotics: {self.num_known_ab}")
        elif self.masking_method == 'num_known_classes':
            print(f"Number of known classes: {self.num_known_classes}")
        print(f"Number of epochs: {self.epochs}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Loss function: {'BCE' if self.loss_fn == 'bce' else 'Focal'}")
        if self.loss_fn == 'focal':
            print(f"Gamma: {self.gamma}")
        print(f"Learning rate: {self.lr}")
        print(f"Weight decay: {self.weight_decay}")
        print("="*self._splitter_size)
    
    def __call__(self):      
        assert self.model.pheno_only == True, "Model must be in pheno_only mode"
        if not self.CV_mode:
            self.wandb_run = self._init_wandb()
        print("Initializing training...")
        self.val_set.prepare_dataset()
        self.val_loader = DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False)
        
        start_time = time.time()
        self.best_val_loss = float('inf') 
        self._init_result_lists()
        for self.current_epoch in range(self.current_epoch, self.epochs):
            self.model.train()
            self.train_set.prepare_dataset()
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            epoch_start_time = time.time()
            train_loss = self.train(self.current_epoch) # returns loss, averaged over batches
            self.losses.append(train_loss)
            if time.time() - epoch_start_time > 60:
                disp_time = f"{(time.time() - epoch_start_time)/60:.1f} min"
            else:
                disp_time = f"{time.time() - epoch_start_time:.0f} sec"
            print(f"Epoch completed in " + disp_time + f" | Loss: {train_loss:.4f}")
            val_start = time.time()
            print("Evaluating on validation set...")
            val_results = self.evaluate(self.val_loader, self.val_set)
            if time.time() - val_start > 60:
                disp_time = f"{(time.time() - val_start)/60:.1f} min"
            else:
                disp_time = f"{time.time() - val_start:.0f} sec"
            print(f"Validation completed in " + disp_time)
            s = f"Val loss: {val_results['loss']:.4f}"
            s += f" | Accuracy {val_results['acc']:.2%} | Isolate accuracy {val_results['iso_acc']:.2%}"
            print(s)
            self._update_val_lists(val_results)
            if not self.CV_mode:
                self._report_epoch_results()
            early_stop = self.early_stopping()
            print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")
            print("="*self._splitter_size)
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            if early_stop:
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.4f}")
                print(f"Validation stats at best epoch ({self.best_epoch+1}):")
                s = f"Loss: {self.val_losses[self.best_epoch]:.4f}" 
                s += f" | Accuracy: {self.val_accs[self.best_epoch]:.2%}"
                s += f" | Isolate accuracy: {self.val_iso_accs[self.best_epoch]:.2%}"
                print(s)
                if not self.CV_mode:
                    self.wandb_run.log({
                        "Losses/final_val_loss": self.best_val_loss, 
                        "Accuracies/final_val_acc": self.val_accs[self.best_epoch],
                        "Accuracies/final_val_iso_acc": self.val_iso_accs[self.best_epoch],
                        "Class_metrics/final_val_sens": self.val_sensitivities[self.best_epoch],
                        "Class_metrics/final_val_spec": self.val_specificities[self.best_epoch],
                        "Class_metrics/final_val_F1": self.val_F1_scores[self.best_epoch],
                        "Class_metrics/final_val_auc_score": self.val_auc_scores[self.best_epoch],
                        "best_epoch": self.best_epoch+1
                    })
                self.model.load_state_dict(self.best_model_state) 
                self.current_epoch = self.best_epoch
                break
            if self.scheduler:
                self.scheduler.step()
        if not early_stop and not self.CV_mode: 
            self.wandb_run.log({
                    "Losses/final_val_loss": self.best_val_loss, 
                    "Accuracies/final_val_acc": self.val_accs[-1],
                    "Accuracies/final_val_iso_acc": self.val_iso_accs[-1],
                    "Class_metrics/final_val_sens": self.val_sensitivities[self.best_epoch],
                    "Class_metrics/final_val_spec": self.val_specificities[self.best_epoch],
                    "Class_metrics/final_val_F1": self.val_F1_scores[self.best_epoch],
                    "Class_metrics/final_val_auc_score": self.val_auc_scores[self.best_epoch],
                    "best_epoch": self.current_epoch+1
                })
        if self.save_model_:
            self.save_model(self.results_dir / "model_state.pt") 
        train_time = (time.time() - start_time)/60
        if not self.CV_mode:
            self.wandb_run.log({"Training time (min)": train_time})
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        print("="*self._splitter_size)
        if not early_stop:
            print("Final validation stats:")
            s = f"Loss: {self.val_losses[-1]:.4f}"
            s += f" | Accuracy: {self.val_accs[-1]:.2%}"
            s += f" | Isolate accuracy: {self.val_iso_accs[-1]:.2%}"
            print(s)
        
        results = {
            "train_loss": self.losses[self.best_epoch],
            "loss": self.val_losses[self.best_epoch],
            "acc": self.val_accs[self.best_epoch],
            "iso_acc": self.val_iso_accs[self.best_epoch],
            "sens": self.val_sensitivities[self.best_epoch],
            "spec": self.val_specificities[self.best_epoch],
            "prec": self.val_precisions[self.best_epoch],
            "F1": self.val_F1_scores[self.best_epoch],
            "auc_score": self.val_auc_scores[self.best_epoch],
            "roc": self.val_roc_results[self.best_epoch],
            "iso_stats": self.val_iso_stats[self.best_epoch],
            "ab_stats": self.val_ab_stats[self.best_epoch]
        }
        return results
    
    
    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_loss, reporting_loss, printing_loss = 0, 0, 0
        for i, batch in enumerate(self.train_loader):
            batch_index = i + 1
            self.optimizer.zero_grad() # zero out gradients
            
            input, target_res, _, token_types, attn_mask = batch
            pred_logits = self.model(input, token_types, attn_mask) # get predictions for all antibiotics
            ab_mask = target_res != -1 # (batch_size, num_ab), True if antibiotic is masked, False otherwise
            
            ab_indices = ab_mask.any(dim=0).nonzero().squeeze(-1).tolist() # list of indices of antibiotics present in the batch
            losses = list()
            for j in ab_indices: 
                mask = ab_mask[:, j] # (batch_size,), indicates which samples contain the antibiotic masked
                # isolate the predictions and targets for the antibiotic
                ab_pred_logits = pred_logits[mask, j] # (num_masked_samples,)
                ab_targets = target_res[mask, j] # (num_masked_samples,)
                ab_loss = self.ab_criterions[j](ab_pred_logits, ab_targets)
                losses.append(ab_loss)
            loss = sum(losses) / len(losses) # average loss over antibiotics
            epoch_loss += loss.item()
            reporting_loss += loss.item()
            printing_loss += loss.item()
            
            loss.backward() 
            self.optimizer.step() 
            if batch_index % self.report_every == 0:
                if not self.CV_mode:
                    self._report_loss_results(batch_index, reporting_loss)
                reporting_loss = 0 
                
            if batch_index % self.print_progress_every == 0:
                time_elapsed = time.gmtime(time.time() - time_ref) 
                self._print_loss_summary(time_elapsed, batch_index, printing_loss) 
                printing_loss = 0  
        avg_epoch_loss = epoch_loss / self.num_batches
        return avg_epoch_loss 
    
    
    def early_stopping(self):
        if self.current_epoch > 4: # excempt the first 5 epochs from early stopping
            if self.val_losses[-1] < self.best_val_loss:
                self.best_val_loss = self.val_losses[-1]
                self.best_epoch = self.current_epoch
                self.best_model_state = self.model.state_dict()
                self.early_stopping_counter = 0
                return False
            else:
                self.early_stopping_counter += 1
                return True if self.early_stopping_counter >= self.patience else False
        else:
            self.early_stopping_counter = 0
            return False
        
            
    def evaluate(self, loader: DataLoader, ds_obj):
        self.model.eval()
        # prepare evaluation statistics dataframes
        ab_stats, iso_stats = self._init_eval_stats(ds_obj)
        with torch.no_grad(): 
            ## Antibiotic tracking ##
            ab_num = np.zeros((self.num_ab, 2)) # tracks the occurence for each antibiotic & resistance
            ab_num_preds = np.zeros_like(ab_num) # tracks the number of predictions for each antibiotic & resistance
            ab_num_correct = np.zeros_like(ab_num) # tracks the number of correct predictions for each antibiotic & resistance
            ## General tracking ##
            pred_sigmoids = torch.tensor([]).to(device)
            target_resistances = torch.tensor([]).to(device)
            loss = 0
            for i, batch in enumerate(loader):                  
                input, target_res, target_ids, token_types, attn_mask = batch
                pred_logits = self.model(input, token_types, attn_mask) # get predictions for all antibiotics
                
                ###### ROC ######
                pred_sigmoids = torch.cat((pred_sigmoids, torch.sigmoid(pred_logits)), dim=0) # (+batch_size, num_ab)
                target_resistances = torch.cat((target_resistances, target_res), dim=0)
                
                pred_res = torch.where(pred_logits > 0, torch.ones_like(pred_logits), torch.zeros_like(pred_logits)) # logits -> 0/1 (S/R)
                ab_mask = target_res >= 0 # (batch_size, num_ab), True if antibiotic is masked, False otherwise
                iso_stats = self._update_iso_stats(i, pred_res, target_res, target_ids, ab_mask, token_types, iso_stats) 
                
                ab_indices = ab_mask.any(dim=0).nonzero().squeeze(-1).tolist() # list of indices of antibiotics present in the batch
                losses = list()
                for j in ab_indices: 
                    mask = ab_mask[:, j] # (batch_size,)
                    
                    # isolate the predictions and targets for the antibiotic
                    ab_pred_logits = pred_logits[mask, j] # (num_masked_samples,)
                    ab_targets = target_res[mask, j] # (num_masked_samples,)
                    num_R = ab_targets.sum().item()
                    num_S = ab_targets.shape[0] - num_R
                    ab_num[j, :] += [num_S, num_R]
                    
                    ab_loss = self.ab_criterions[j](ab_pred_logits, ab_targets)
                    losses.append(ab_loss)
                    
                    ab_pred_res = pred_res[mask, j]
                    ab_num_correct[j, :] += self._get_num_correct(ab_pred_res, ab_targets)    
                    ab_num_preds[j, :] += self._get_num_preds(ab_pred_res)
                loss += sum(losses) / len(losses) # average loss over antibiotics
                    
            avg_loss = loss.item() / len(loader)
            pred_sigmoids = pred_sigmoids.cpu().numpy()
            target_resistances = target_resistances.cpu().numpy()
            roc_results = self._get_roc_results(pred_sigmoids, target_resistances)
            
            ab_stats = self._update_ab_eval_stats(ab_stats, ab_num, ab_num_preds, ab_num_correct, 
                                                  pred_sigmoids, target_resistances)
            iso_stats = self._calculate_iso_stats(iso_stats)
        
            acc = ab_stats['num_correct'].sum() / ab_stats['num_masked_tot'].sum()
            iso_acc = iso_stats['all_correct'].sum() / iso_stats.shape[0]
            sens = ab_stats['num_correct_R'].sum() / ab_stats['num_masked_R'].sum() 
            spec = ab_stats['num_correct_S'].sum() / ab_stats['num_masked_S'].sum()
            prec = ab_stats['num_correct_R'].sum() / ab_stats['num_pred_R'].sum()
            F1_score = 2 * sens * prec / (sens + prec)

            results = {
                "loss": avg_loss, 
                "acc": acc,
                "iso_acc": iso_acc,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "F1": F1_score,
                "ab_stats": ab_stats,
                "iso_stats": iso_stats,
                "roc_results": roc_results,
                "auc_score": roc_results["auc_score"]
            }
        return results
            
    
    def _init_result_lists(self):
        self.losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_iso_accs = []
        self.val_sensitivities = []
        self.val_specificities = []
        self.val_precisions = []
        self.val_F1_scores = []
        self.val_ab_stats = []
        self.val_iso_stats = []
        self.val_roc_results = []
        self.val_auc_scores = []
        
        
    def _update_val_lists(self, results: dict):
        self.val_losses.append(results["loss"])
        self.val_accs.append(results["acc"])
        self.val_iso_accs.append(results["iso_acc"])
        self.val_sensitivities.append(results["sensitivity"])
        self.val_specificities.append(results["specificity"])
        self.val_precisions.append(results["precision"])
        self.val_F1_scores.append(results["F1"])
        self.val_ab_stats.append(results["ab_stats"])
        self.val_iso_stats.append(results["iso_stats"])
        self.val_roc_results.append(results["roc_results"])
        self.val_auc_scores.append(results["auc_score"])
    
    
    def _init_eval_stats(self, ds_obj):
        ab_stats = pd.DataFrame(columns=[
            'antibiotic', 'num_masked_tot', 'num_masked_S', 'num_masked_R', 'num_pred_S', 'num_pred_R', 
            'num_correct', 'num_correct_S', 'num_correct_R', 'accuracy', 'sensitivity', 'specificity', 'precision', 'F1'
        ])
        ab_stats['antibiotic'] = self.antibiotics
        ab_stats['num_masked_tot'], ab_stats['num_masked_S'], ab_stats['num_masked_R'] = 0, 0, 0
        ab_stats['num_pred_S'], ab_stats['num_pred_R'] = 0, 0
        ab_stats['num_correct'], ab_stats['num_correct_S'], ab_stats['num_correct_R'] = 0, 0, 0
        ab_stats['auc_score'] = 0.0
        ab_stats['roc_fpr'], ab_stats['roc_tpr'], ab_stats['roc_thresholds'] = None, None, None
        
        iso_stats = ds_obj.ds.copy()
        iso_stats['num_masked_ab'], iso_stats['num_masked_genes'] = 0, 0 
        iso_stats['num_masked_S'], iso_stats['num_masked_R'] = 0, 0
        iso_stats['num_correct'], iso_stats['correct_S'], iso_stats['correct_R'] = 0, 0, 0
        iso_stats['sensitivity'], iso_stats['specificity'], iso_stats['accuracy'] = 0, 0, 0
        iso_stats['masked_ab'], iso_stats['correct_ab'], iso_stats['masked_genes'] = None, None, None
        iso_stats['all_correct'] = False  
        return ab_stats, iso_stats
    
    
    def _update_ab_eval_stats(self, ab_stats: pd.DataFrame, num, num_preds, num_correct, 
                              pred_sigmoids: np.ndarray, target_resistances: np.ndarray):
        for j in range(self.num_ab): 
            target_resistances_ab, pred_sigmoids_ab = target_resistances[:, j], pred_sigmoids[:, j]
            pred_sigmoids_ab = pred_sigmoids_ab[target_resistances_ab >= 0]
            target_resistances_ab = target_resistances_ab[target_resistances_ab >= 0]
            try:
                if len(np.unique(target_resistances_ab)) == 1:
                    auc_score_ab = np.nan
                    raise ValueError(f"Only one class present for antibiotic {self.antibiotics[j]}. AUC score is undefined.")
                
                auc_score_ab = roc_auc_score(target_resistances_ab, pred_sigmoids_ab)
                fpr, tpr, thresholds = roc_curve(target_resistances_ab, pred_sigmoids_ab)
            except ValueError as e:
                print(e)
                auc_score_ab = np.nan              
            ab_stats.at[j, 'roc_fpr'], ab_stats.at[j, 'roc_tpr'], ab_stats.at[j, 'roc_thresholds'] = fpr, tpr, thresholds   
            ab_stats.loc[j, 'auc_score'] = auc_score_ab
            ab_stats.loc[j, 'num_masked_tot'] = num[j, :].sum()
            ab_stats.loc[j, 'num_masked_S'], ab_stats.loc[j, 'num_masked_R'] = num[j, 0], num[j, 1]
            ab_stats.loc[j, 'num_pred_S'], ab_stats.loc[j, 'num_pred_R'] = num_preds[j, 0], num_preds[j, 1]
            ab_stats.loc[j, 'num_correct'] = num_correct[j, :].sum()
            ab_stats.loc[j, 'num_correct_S'], ab_stats.loc[j, 'num_correct_R'] = num_correct[j, 0], num_correct[j, 1]
        ab_stats['accuracy'] = ab_stats.apply(
            lambda row: row['num_correct']/row['num_masked_tot'] if row['num_masked_tot'] > 0 else np.nan, axis=1)
        ab_stats['sensitivity'] = ab_stats.apply(
            lambda row: row['num_correct_R']/row['num_masked_R'] if row['num_masked_R'] > 0 else np.nan, axis=1)
        ab_stats['specificity'] = ab_stats.apply(
            lambda row: row['num_correct_S']/row['num_masked_S'] if row['num_masked_S'] > 0 else np.nan, axis=1)
        ab_stats['precision'] = ab_stats.apply(
            lambda row: row['num_correct_R']/row['num_pred_R'] if row['num_pred_R'] > 0 else np.nan, axis=1)
        ab_stats['F1'] = ab_stats.apply(
            lambda row: 2*row['precision']*row['sensitivity']/(row['precision']+row['sensitivity']) 
            if row['precision'] > 0 and row['sensitivity'] > 0 else np.nan, axis=1)
        return ab_stats
    
    
    def _get_num_correct(self, pred_res: torch.Tensor, target_res: torch.Tensor):
        eq = torch.eq(pred_res, target_res)
        num_correct_S = eq[target_res == 0].sum().item()
        num_correct_R = eq[target_res == 1].sum().item()
        return [num_correct_S, num_correct_R]
    
    
    def _get_num_preds(self, pred_res: torch.Tensor):
        num_pred_S = (pred_res == 0).sum().item()
        num_pred_R = (pred_res == 1).sum().item()
        return [num_pred_S, num_pred_R]
    
    
    def _get_roc_results(self, pred_sigmoids: np.ndarray, target_resistances: np.ndarray, drop_intermediate: bool = False):
        pred_sigmoids_flat = pred_sigmoids[target_resistances >= 0]
        target_resistances_flat = target_resistances[target_resistances >= 0]
        assert pred_sigmoids_flat.shape == target_resistances_flat.shape, "Shapes do not match"
        assert len(pred_sigmoids_flat.shape) == 1, "Only 1D arrays are supported"
        
        fpr, tpr, thresholds = roc_curve(target_resistances_flat, pred_sigmoids_flat, drop_intermediate=drop_intermediate)
        auc_score = auc(fpr, tpr)
        roc_results = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc_score": auc_score}
        
        return roc_results
    
    
    def _update_iso_stats(self, batch_idx, pred_res: torch.Tensor, target_res: torch.Tensor, target_ids: torch.Tensor,
                          ab_mask: torch.Tensor, token_types:torch.tensor, iso_stats: pd.DataFrame):
        for i in range(pred_res.shape[0]): 
            iso_ab_mask = ab_mask[i]
            iso_token_types = token_types[i][target_ids[i] != -1] # token types masked tokens
            iso_target_ids = target_ids[i][target_ids[i] != -1] # token ids of the antibiotics and genes that are masked
            df_idx = batch_idx * self.val_batch_size + i # index of the isolate in the combined dataset
            
            # counts
            num_masked_ab = iso_ab_mask.sum().item()
            num_masked_R = target_res[i][iso_ab_mask].sum().item()
            num_masked_S = num_masked_ab - num_masked_R
            
            # statistics
            masked_ab_indices = iso_ab_mask.nonzero().squeeze(-1).tolist() # ab-indexing index of the masked antibiotics 
            iso_target_res = target_res[i][iso_ab_mask] # (num_masked_ab,)
            eq = torch.eq(pred_res[i][iso_ab_mask], iso_target_res) # (num_masked_ab,)
            num_correct_R = eq[iso_target_res == 1].sum().item()
            num_correct_S = eq[iso_target_res == 0].sum().item()
            num_correct = num_correct_S + num_correct_R
            all_correct = eq.all().item()
            
            # add masked genes and antibiotics
            ab_indices = iso_target_ids[iso_token_types == 2].tolist() # token ids of the masked antibiotics, sequence order
            masked_ab = [self.vocab.lookup_token(idx) for idx in ab_indices] # token, sequence order
            masked_ab_indices_seq = [self.ab_to_idx[token.split('_')[0]] for token in masked_ab] # index in the ab-indexing, sequence order
            correct_ab = [eq[masked_ab_indices.index(idx)].item() for idx in masked_ab_indices_seq]
            
            geno_indices = iso_target_ids[iso_token_types == 1].tolist()
            masked_genes = [self.vocab.lookup_token(idx) for idx in geno_indices]
            
            data = {
                'num_masked_genes': len(geno_indices), 'masked_genes': masked_genes, 
                'masked_ab': pd.Series(masked_ab).tolist(), 'correct_ab': correct_ab,
                'num_masked_ab': num_masked_ab, 'num_masked_S': num_masked_S, 'num_masked_R': num_masked_R, 
                'num_correct': num_correct, 'correct_S': num_correct_S, 'correct_R': num_correct_R,
                'all_correct': all_correct
            }
            iso_stats.loc[df_idx, data.keys()] = data.values()
        return iso_stats
    
    def _calculate_iso_stats(self, iso_stats: pd.DataFrame): 
        iso_stats['accuracy'] = iso_stats['num_correct'] / iso_stats['num_masked_ab']
        iso_stats['sensitivity'] = iso_stats.apply(
            lambda row: row['correct_R']/row['num_masked_R'] if row['num_masked_R'] > 0 else np.nan, axis=1
        )
        iso_stats['specificity'] = iso_stats.apply(
            lambda row: row['correct_S']/row['num_masked_S'] if row['num_masked_S'] > 0 else np.nan, axis=1
        )
        return iso_stats
        
     
    def _init_wandb(self):
        self.wandb_run = wandb.init(
            project=self.project_name, # name of the project
            name=self.wandb_name, # name of the run
            
            config={
                "trainer_type": "fine-tuning",
                "Device:" : f"{device.type} ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else device.type,
                "exp_folder": self.exp_folder,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "emb_dim": self.model.emb_dim,
                'ff_dim': self.model.ff_dim,
                "lr": self.lr,
                "loss_fn": self.loss_fn,
                "ab_weights": self.ab_weights if self.wl_strength else None,
                "weight_decay": self.weight_decay,
                "masking_method": self.masking_method, 
                "mask_prob_geno": self.mask_prob_geno,
                "mask_prob_pheno": self.mask_prob_pheno,
                "num_known_ab": self.num_known_ab,
                "max_seq_len": self.model.max_seq_len,
                "vocab_size": len(self.vocab),
                "num_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "num_antibiotics": self.num_ab,
                "antibiotics": self.antibiotics,
                "train_share": round(self.train_share, 2),
                "train_size": self.train_size,
                "random_state": self.random_state,
                "CV_mode": self.CV_mode,
                'val_share': round(self.val_share, 2),
                "val_size": self.val_size,
                "is_pretrained": self.model.is_pretrained,
            }
        )
        self.wandb_run.watch(self.model) # watch the model for gradients and parameters
        self.wandb_run.define_metric("epoch", hidden=True)
        self.wandb_run.define_metric("batch", hidden=True)
        
        self.wandb_run.define_metric("Losses/live_loss", step_metric="batch")
        self.wandb_run.define_metric("Losses/train_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Losses/val_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_acc", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_iso_acc", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Class_metrics/val_sens", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Class_metrics/val_spec", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Class_metrics/val_F1", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Class_metrics/val_auc_score", summary="max", step_metric="epoch")
        
        self.wandb_run.define_metric("Losses/final_val_loss")
        self.wandb_run.define_metric("Accuracies/final_val_acc")
        self.wandb_run.define_metric("Accuracies/final_val_iso_acc")
        self.wandb_run.define_metric("Class_metrics/final_val_sens")
        self.wandb_run.define_metric("Class_metrics/final_val_spec")
        self.wandb_run.define_metric("Class_metrics/final_val_F1")
        self.wandb_run.define_metric("Class_metrics/final_val_auc_score")
        
        self.wandb_run.define_metric("best_epoch", hidden=True)

        return self.wandb_run
     
    def _report_epoch_results(self):
        wandb_dict = {
            "epoch": self.current_epoch+1,
            "Losses/train_loss": self.losses[-1],
            "Losses/val_loss": self.val_losses[-1],
            "Losses/val_loss": self.val_losses[-1],
            "Class_metrics/val_sens": self.val_sensitivities[-1],
            "Class_metrics/val_spec": self.val_specificities[-1],
            "Class_metrics/val_F1": self.val_F1_scores[-1],
            "Class_metrics/val_auc_score": self.val_auc_scores[-1],
            "Accuracies/val_acc": self.val_accs[-1],
            "Accuracies/val_iso_acc": self.val_iso_accs[-1],
        }
        self.wandb_run.log(wandb_dict)
    
        
    def _report_loss_results(self, batch_index, tot_loss):
        avg_loss = tot_loss / self.report_every
        
        global_step = self.current_epoch * self.num_batches + batch_index # global step, total #batches seen
        self.wandb_run.log({"batch": global_step, "Losses/live_loss": avg_loss})
    
        
    def _print_loss_summary(self, time_elapsed, batch_index, tot_loss):
        progress = batch_index / self.num_batches
        mlm_loss = tot_loss / self.print_progress_every
          
        s = f"{time.strftime('%H:%M:%S', time_elapsed)}" 
        s += f" | {batch_index}/{self.num_batches} ({progress:.2%}) | "\
                f"Loss: {mlm_loss:.4f}"
        print(s)
    
    
    def save_model(self, savepath: Path = None):
        if not savepath:
            savepath = self.results_dir / "model_state.pt"
        torch.save(self.best_model_state, savepath)
        print(f"Model saved to {savepath}")
        print("="*self._splitter_size)
        
        
    def load_model(self, savepath: Path):
        print("="*self._splitter_size)
        print(f"Loading model from {savepath}")
        self.model.load_state_dict(torch.load(savepath))
        self.model.to(device)
        print("Model loaded")
 