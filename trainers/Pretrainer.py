import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time 
import wandb

from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

from utils import WeightedBCEWithLogitsLoss, BinaryFocalWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################################
############################################### PRE-TRAINING TRAINER ###################################################
########################################################################################################################

class MMBertPreTrainer(nn.Module):
    
    def __init__(self,
                 config: dict,
                 model,
                 antibiotics: list, # list of antibiotics in the dataset
                 train_set,
                 val_set,
                 results_dir: Path = None,
                 ):
        super(MMBertPreTrainer, self).__init__()
        
        self.random_state = config["random_state"]
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True 
        
        self.model = model
        self.project_name = config["project_name"]
        self.exp_folder = config["exp_folder"]
        self.wandb_name = config["name"] if config["name"] else datetime.now().strftime("%Y%m%d-%H%M%S")
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics)
        
        self.train_set, self.train_size = train_set, len(train_set)
        self.val_set, self.val_size = val_set, len(val_set) 
        assert round(self.val_size / (self.train_size + self.val_size), 2) == config["val_share"], "Validation set size does not match intended val_share"
        self.val_share, self.train_share = config["val_share"], 1 - config["val_share"]
        self.batch_size = config["batch_size"]
        self.val_batch_size = 16*self.batch_size
        self.num_batches = np.ceil(self.train_size / self.batch_size).astype(int)
        self.vocab = self.train_set.vocab
         
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.patience = config["early_stopping_patience"]
        self.save_model_ = config["save_model"]
        self.do_eval = config["do_eval"] 
        
        self.mask_prob_geno = self.train_set.mask_prob_geno
        self.mask_prob_pheno = self.train_set.mask_prob_pheno
        self.num_known_ab = self.train_set.num_known_ab
        self.num_known_classes = self.train_set.num_known_classes
        
        self.geno_criterion = nn.CrossEntropyLoss(ignore_index = -1).to(device) # ignores loss where target_ids == -1
        self.loss_fn = config["loss_fn"]
        self.gamma = config["gamma"]  ## hyperparameters for focal loss
        self.wl_strength = config["wl_strength"]  ## positive class weight for weighted BCELoss
        if self.wl_strength:
            self.ab_weights = config['data']['antibiotics']['ab_weights_'+self.wl_strength]
            self.ab_weights = {ab: v for ab, v in self.ab_weights.items() if ab in self.antibiotics}
            self.alphas = [v for v in self.ab_weights.values()]
        else:
            self.alphas = [0.5]*self.num_ab         ## equal class weights for all antibiotics
            
        if self.loss_fn == 'bce':
            self.ab_criterions = [WeightedBCEWithLogitsLoss(alpha=alpha).to(device) for alpha in self.alphas]
        elif self.loss_fn == 'focal':                   ## can implement antibiotic-specific parameters
            self.ab_criterions = [BinaryFocalWithLogitsLoss(alpha, self.gamma).to(device) for alpha in self.alphas]
        else:
            raise NotImplementedError("Only 'bce' and 'focal' functions are supported")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
                 
        self.current_epoch = 0
        self.report_every = config["report_every"]
        self.print_progress_every = config["print_progress_every"]
        self._splitter_size = 80
        self.results_dir = results_dir
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True) 
        
        
    def print_model_summary(self):        
        print("Model summary:")
        print("="*self._splitter_size)
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
        print(f"CV split: {self.train_share:.0%} train | {self.val_share:.0%} val")
        print(f"Eval mode: {'on' if self.do_eval else 'off'}")
        print(f"Mask probability (genotypes): {self.mask_prob_geno:.0%}")
        print(f"Masking method: {self.train_set.masking_method}")
        if self.mask_prob_pheno:
            print(f"Mask probability (phenotypes): {self.mask_prob_pheno:.0%}")
        if self.num_known_ab:
            print(f"Number of known antibiotics: {self.num_known_ab}")
        if self.num_known_classes:
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
        self._init_wandb()
        print("="*self._splitter_size)
        print("Preparing validation set...")
        self.val_set.prepare_dataset()
        self.val_set.shuffle() # to avoid batches of only genotypes or only phenotypes
        self.val_loader = DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False)
        print("Initializing training...")
        
        start_time = time.time()
        self.best_val_loss = float('inf') 
        self._init_result_lists()
        for self.current_epoch in range(self.current_epoch, self.epochs):
            self.model.train()
            # Dynamic masking: New mask for training set each epoch
            self.train_set.prepare_dataset()
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            epoch_start_time = time.time()
            train_losses = self.train(self.current_epoch) # returns loss, averaged over batches
            self.losses.append(train_losses['loss'])
            self.pheno_losses.append(train_losses['pheno_loss'])
            self.geno_losses.append(train_losses['geno_loss']) 
            print(f"Epoch completed in {(time.time() - epoch_start_time)/60:.1f} min")
            print("Loss: {:.4f} | Genotype loss: {:.4f} | Phenotype loss: {:.4f}".format(
                train_losses['loss'], train_losses['geno_loss'], train_losses['pheno_loss']))
            if self.do_eval:
                print("Evaluating on validation set...")
                val_start = time.time()
                val_results = self.evaluate(self.val_loader, self.val_set)
                if time.time() - val_start > 60:
                    disp_time = f"{(time.time() - val_start)/60:.1f} min"
                else:
                    disp_time = f"{time.time() - val_start:.0f} sec"
                print(f"Validation completed in " + disp_time)
                self.print_val_results(val_results)
                self._update_val_lists(val_results)
            else:
                val_losses = self.get_val_loss(self.val_loader)
                self.val_losses.append(val_losses['loss'])
                self.val_geno_losses.append(val_losses['geno_loss'])
                self.val_pheno_losses.append(val_losses['pheno_loss'])
                print(f"Validation loss: {val_losses['loss']:.4f}")
                print(f"Genotype loss: {val_losses['geno_loss']:.4f} | Phenotype loss: {val_losses['pheno_loss']:.4f}")
            self._report_epoch_results()
            early_stop = self.early_stopping()
            print(f"Early stopping counter: {self.early_stopping_counter}/{self.patience}")
            print("="*self._splitter_size)
            num_days = (time.time() - start_time) // (24*60*60)
            print(f"Elapsed time: {int(num_days):02d}-{time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            if early_stop:
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.4f}")
                if self.do_eval:
                    self.print_early_stop_results()
                    self.report_early_stop_results()
                else:
                    wandb_dict = {
                        "Losses/final_val_loss": self.val_losses[self.best_epoch],
                        "Losses/final_val_geno_loss": self.val_geno_losses[self.best_epoch],
                        "Losses/final_val_pheno_loss": self.val_pheno_losses[self.best_epoch],
                        "best_epoch": self.best_epoch+1
                    }
                    self.wandb_run.log(wandb_dict)
                print("="*self._splitter_size)
                self.model.load_state_dict(self.best_model_state)
                self.current_epoch = self.best_epoch
                break
            if self.scheduler:
                self.scheduler.step()
        if not early_stop:
            if self.do_eval:    
                self.wandb_run.log({
                        "Losses/final_val_loss": self.best_val_loss, 
                        "Losses/final_val_geno_loss": self.val_geno_losses[-1],
                        "Losses/final_val_pheno_loss": self.val_pheno_losses[-1],
                        "Accuracies/final_val_pheno_acc": self.val_pheno_accs[-1],
                        "Accuracies/final_val_pheno_iso_acc": self.val_pheno_iso_accs[-1],
                        "Accuracies/final_val_geno_acc": self.val_geno_accs[-1],
                        "Accuracies/final_val_geno_iso_acc": self.val_geno_iso_accs[-1],
                        "best_epoch": self.current_epoch+1
                    })
            else:
                wandb_dict = {
                    "Losses/final_val_loss": self.val_losses[-1],
                    "Losses/final_val_geno_loss": self.val_geno_losses[-1],
                    "Losses/final_val_pheno_loss": self.val_pheno_losses[-1],
                    "best_epoch": self.current_epoch+1
                }
                self.wandb_run.log(wandb_dict)
        if self.save_model_:
            self.save_model() 
        train_time = (time.time() - start_time)/60
        self.wandb_run.log({"Training time (min)": train_time})
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        print("="*self._splitter_size)
        if not early_stop and self.do_eval:
            print("Final validation stats:")
            s1 = f"Loss: {self.val_losses[-1]:.4f} | Phenotype Loss: {self.val_pheno_losses[-1]:.4f}"
            s1 += f" | Genotype Loss: {self.val_geno_losses[-1]:.4f}"
            print(s1)
            s2 = f"Phenotype accuracy: {self.val_pheno_accs[-1]:.2%}"
            s2 += f" | Phenotype isolate accuracy: {self.val_pheno_iso_accs[-1]:.2%}"
            print(s2)
            s3 = f" Genotype accuracy: {self.val_geno_accs[-1]:.2%}"
            s3 += f" | Genotype isolate accuracy: {self.val_geno_iso_accs[-1]:.2%}"
            print(s3)
        if self.do_eval:
            results = {
                "best_epoch": self.current_epoch,
                "train_losses": self.losses,
                "val_losses": self.val_losses,
                "val_pheno_losses": self.val_pheno_losses,
                "val_geno_losses": self.val_geno_losses,
                "val_pheno_accs": self.val_pheno_accs,
                "val_geno_accs": self.val_geno_accs,
                "val_pheno_iso_accs": self.val_pheno_iso_accs,
                "val_geno_iso_accs": self.val_geno_iso_accs,
                "train_time": train_time,
                "val_iso_stats_geno": self.val_iso_stats_geno[self.best_epoch],
                "val_iso_stats_pheno": self.val_iso_stats_pheno[self.best_epoch],
                "val_ab_stats": self.val_ab_stats[self.best_epoch]
            }
        else:
            results = {
                "best_epoch": self.current_epoch,
                "train_losses": self.losses,
                "train_time": train_time,
                "val_losses": self.val_losses
            }
        return results
    
    
    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_geno_loss, epoch_pheno_loss = 0, 0
        geno_batches, pheno_batches = 0, 0
        reporting_loss, printing_loss = 0, 0
        for i, batch in enumerate(self.train_loader):
            batch_index = i + 1
            self.optimizer.zero_grad() # zero out gradients
            
            input, target_ids, target_res, token_types, attn_mask = batch   
            # input, target_ids, target_res, token_types, attn_mask, masked_sequences = batch   
            pred_logits, token_pred = self.model(input, token_types, attn_mask) # get predictions for all antibiotics
            ab_mask = target_res != -1 # (batch_size, num_ab), True if antibiotic is masked, False otherwise
            
            loss = 0
            if ab_mask.any(): # if there are phenotypes in the batch
                ## Phenotype loss ##
                ab_indices = ab_mask.any(dim=0).nonzero().squeeze(-1).tolist() # list of indices of antibiotics present in the batch
                losses = list()
                for j in ab_indices: 
                    mask = ab_mask[:, j] # (batch_size,), indicates which samples contain the antibiotic masked
                    # isolate the predictions and targets for the antibiotic
                    ab_pred_logits = pred_logits[mask, j] # (num_masked_samples,)
                    ab_targets = target_res[mask, j] # (num_masked_samples,)
                    ab_loss = self.ab_criterions[j](ab_pred_logits, ab_targets)
                    losses.append(ab_loss)
                pheno_loss = sum(losses) / len(losses) # average loss over antibiotics
                epoch_pheno_loss += pheno_loss.item()
                pheno_batches += 1
                loss += pheno_loss
                
            if (target_ids != -1).any(): # if there are genotypes in the batch
                ## Genotype loss ##
                geno_loss = self.geno_criterion(token_pred.transpose(-1, -2), target_ids)
                epoch_geno_loss += geno_loss.item()
                geno_batches += 1
                loss += geno_loss
            reporting_loss += loss.item()
            printing_loss += loss.item()
            
            loss.backward() 
            self.optimizer.step() 
            if self.report_every:
                if batch_index % self.report_every == 0:
                    self._report_loss_results(batch_index, reporting_loss)
                    reporting_loss = 0 
                
            if batch_index % self.print_progress_every == 0:
                time_elapsed = time.gmtime(time.time() - time_ref) 
                self._print_loss_summary(time_elapsed, batch_index, printing_loss) 
                printing_loss = 0  
        avg_pheno_loss = epoch_pheno_loss / pheno_batches
        avg_geno_loss = epoch_geno_loss / geno_batches
        avg_epoch_loss = avg_geno_loss + avg_pheno_loss
        losses = {"loss": avg_epoch_loss, "geno_loss": avg_geno_loss, "pheno_loss": avg_pheno_loss}
        return losses 
    
    
    def early_stopping(self):
        if self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            self.best_epoch = self.current_epoch
            self.best_model_state = self.model.state_dict()
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return True if self.early_stopping_counter >= self.patience else False
        
    
    def get_val_loss(self, loader: DataLoader):
        self.model.eval()
        with torch.no_grad():
            tot_geno_loss, tot_pheno_loss = 0, 0
            geno_batches, pheno_batches = 0, 0
            for batch in loader:
                input, target_ids, target_res, token_types, attn_mask = batch   
                pred_logits, token_pred = self.model(input, token_types, attn_mask)
                
                ab_mask = target_res >= 0 # (batch_size, num_ab), True if antibiotic is masked, False otherwise
                if ab_mask.any(): # if there are phenotypes in the batch
                    
                    ab_indices = ab_mask.any(dim=0).nonzero().squeeze(-1).tolist() # list of indices of antibiotics present in the batch
                    losses = list()
                    for j in ab_indices: 
                        mask = ab_mask[:, j] # (batch_size,)
                        
                        # isolate the predictions and targets for the antibiotic
                        ab_pred_logits = pred_logits[mask, j] # (num_masked_samples,)
                        ab_targets = target_res[mask, j] # (num_masked_samples,)
                        
                        ab_loss = self.ab_criterions[j](ab_pred_logits, ab_targets)
                        losses.append(ab_loss)
                        
                    pheno_loss = sum(losses) / len(losses) # average loss over antibiotics
                    tot_pheno_loss += pheno_loss.item()
                    pheno_batches += 1
                    
                ###### Genotype loss ######
                token_mask = target_ids != -1 # (batch_size, max_seq_len), True if token is masked, False otherwise
                if token_mask.any(): # if there are genotypes in the batch
                    
                    geno_loss = self.geno_criterion(token_pred.transpose(-1, -2), target_ids) 
                    tot_geno_loss += geno_loss.item()
                    geno_batches += 1
                    
        avg_geno_loss = tot_geno_loss / geno_batches
        avg_pheno_loss = tot_pheno_loss / pheno_batches
        avg_loss = avg_geno_loss + avg_pheno_loss
        val_losses = {"loss": avg_loss, "geno_loss": avg_geno_loss, "pheno_loss": avg_pheno_loss}
        
        return val_losses
    
            
    def evaluate(self, loader: DataLoader, ds_obj):
        self.model.eval()
        # prepare evaluation statistics dataframes
        ab_stats, iso_stats_pheno, iso_stats_geno = self._init_eval_stats(ds_obj)
        with torch.no_grad(): 
            ## Antibiotic tracking ##
            ab_num = np.zeros((self.num_ab, 2)) # tracks the occurence for each antibiotic & resistance
            ab_num_preds = np.zeros_like(ab_num) # tracks the number of predictions for each antibiotic & resistance
            ab_num_correct = np.zeros_like(ab_num) # tracks the number of correct predictions for each antibiotic & resistance
            ## General tracking ##
            tot_pheno_loss, tot_geno_loss = 0, 0
            geno_batches, pheno_batches = 0, 0
            for i, batch in enumerate(loader):                
                input, target_ids, target_res, token_types, attn_mask = batch   
                # input, target_ids, target_res, token_types, attn_mask, sequences, masked_sequences = batch  
                
                pred_logits, token_pred = self.model(input, token_types, attn_mask) # get predictions for all antibiotics
                pred_res = torch.where(pred_logits > 0, torch.ones_like(pred_logits), torch.zeros_like(pred_logits)) # logits -> 0/1 (S/R)
                
                ###### Phenotype loss ######
                ab_mask = target_res >= 0 # (batch_size, num_ab), True if antibiotic is masked, False otherwise
                if ab_mask.any(): # if there are phenotypes in the batch
                    iso_stats_pheno = self._update_pheno_stats(i, pred_res, target_res, ab_mask, iso_stats_pheno)
                    
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
                    pheno_loss = sum(losses) / len(losses) # average loss over antibiotics
                    tot_pheno_loss += pheno_loss.item()
                    pheno_batches += 1
                    
                ###### Genotype loss ######
                token_mask = target_ids != -1 # (batch_size, max_seq_len), True if token is masked, False otherwise
                if token_mask.any(): # if there are genotypes in the batch
                    iso_stats_geno = self._update_geno_stats(i, token_pred, target_ids, token_mask, iso_stats_geno)
                    
                    geno_loss = self.geno_criterion(token_pred.transpose(-1, -2), target_ids) 
                    tot_geno_loss += geno_loss.item()
                    geno_batches += 1
                    
        avg_geno_loss = tot_geno_loss / geno_batches
        avg_pheno_loss = tot_pheno_loss / pheno_batches
        avg_loss = avg_geno_loss + avg_pheno_loss  
        
        ab_stats = self._update_ab_eval_stats(ab_stats, ab_num, ab_num_preds, ab_num_correct)
        iso_stats_geno, iso_stats_pheno = self._calculate_iso_stats(iso_stats_geno, iso_stats_pheno)
        
        pheno_acc = iso_stats_pheno['num_correct'].sum() / iso_stats_pheno['num_masked'].sum()
        pheno_iso_acc = iso_stats_pheno['all_correct'].sum() / iso_stats_pheno.shape[0]
        geno_acc = iso_stats_geno['num_correct'].sum() / iso_stats_geno['num_masked'].sum()
        geno_iso_acc = iso_stats_geno['all_correct'].sum() / iso_stats_geno.shape[0]

        results = {
            "loss": avg_loss, 
            "geno_loss": avg_geno_loss, 
            "pheno_loss": avg_pheno_loss,
            "pheno_acc": pheno_acc,
            "pheno_iso_acc": pheno_iso_acc,
            "geno_acc": geno_acc,
            "geno_iso_acc": geno_iso_acc,
            "ab_stats": ab_stats,
            "iso_stats_pheno": iso_stats_pheno,
            "iso_stats_geno": iso_stats_geno
        }
        return results
            
    
    def _init_result_lists(self):
        self.losses = []
        self.pheno_losses = []
        self.geno_losses = []
        self.val_losses = []
        self.val_geno_losses = []
        self.val_pheno_losses = []
        self.val_pheno_accs = []
        self.val_pheno_iso_accs = []
        self.val_geno_accs = []
        self.val_geno_iso_accs = []
        self.val_ab_stats = []
        self.val_iso_stats_pheno = []
        self.val_iso_stats_geno = []
        
        
    def _update_val_lists(self, results: dict):
        self.val_losses.append(results["loss"])
        self.val_geno_losses.append(results["geno_loss"])
        self.val_pheno_losses.append(results["pheno_loss"])
        self.val_pheno_accs.append(results["pheno_acc"])
        self.val_pheno_iso_accs.append(results["pheno_iso_acc"])
        self.val_geno_accs.append(results["geno_acc"])
        self.val_geno_iso_accs.append(results["geno_iso_acc"])
        self.val_ab_stats.append(results["ab_stats"])
        self.val_iso_stats_pheno.append(results["iso_stats_pheno"])
        self.val_iso_stats_geno.append(results["iso_stats_geno"])
    
    
    def _init_eval_stats(self, ds_obj):
        ab_stats = pd.DataFrame(columns=[
            'antibiotic', 'num_tot', 'num_S', 'num_R', 'num_pred_S', 'num_pred_R', 
            'num_correct', 'num_correct_S', 'num_correct_R',
            'accuracy', 'sensitivity', 'specificity', 'precision', 'F1'
        ])
        ab_stats['antibiotic'] = self.antibiotics
        ab_stats['num_tot'], ab_stats['num_S'], ab_stats['num_R'] = 0, 0, 0
        ab_stats['num_pred_S'], ab_stats['num_pred_R'] = 0, 0
        ab_stats['num_correct'], ab_stats['num_correct_S'], ab_stats['num_correct_R'] = 0, 0, 0

        combined_ds = ds_obj.combined_ds
        ## Extract phenotype samples 
        iso_stats_pheno = combined_ds[combined_ds['source'] == 'pheno'].drop(columns=['source', 'num_genotypes'])
        iso_stats_pheno['num_masked'], iso_stats_pheno['num_masked_S'], iso_stats_pheno['num_masked_R'] = 0, 0, 0
        iso_stats_pheno['num_correct'], iso_stats_pheno['correct_S'], iso_stats_pheno['correct_R'] = 0, 0, 0
        iso_stats_pheno['sensitivity'], iso_stats_pheno['specificity'], iso_stats_pheno['accuracy'] = 0, 0, 0
        iso_stats_pheno['all_correct'] = False 
        
        ## Extract genotype samples
        iso_stats_geno = combined_ds[combined_ds['source'] == 'geno'].drop(columns=['source', 'age', 'gender', 'num_ab'])
        iso_stats_geno.replace(self.val_set.PAD, np.nan, inplace=True)
        iso_stats_geno['num_masked'], iso_stats_geno['num_correct'], iso_stats_geno['accuracy'] = 0, 0, 0
        iso_stats_geno['all_correct'] = False
      
        return ab_stats, iso_stats_pheno, iso_stats_geno
    
    
    def _update_ab_eval_stats(self, ab_stats: pd.DataFrame, num, num_preds, num_correct):
        for j in range(self.num_ab): 
            ab_stats.loc[j, 'num_tot'] = num[j, :].sum()
            ab_stats.loc[j, 'num_S'], ab_stats.loc[j, 'num_R'] = num[j, 0], num[j, 1]
            ab_stats.loc[j, 'num_pred_S'], ab_stats.loc[j, 'num_pred_R'] = num_preds[j, 0], num_preds[j, 1]
            ab_stats.loc[j, 'num_correct'] = num_correct[j, :].sum()
            ab_stats.loc[j, 'num_correct_S'], ab_stats.loc[j, 'num_correct_R'] = num_correct[j, 0], num_correct[j, 1]
        ab_stats['accuracy'] = ab_stats.apply(
            lambda row: row['num_correct']/row['num_tot'] if row['num_tot'] > 0 else np.nan, axis=1)
        ab_stats['sensitivity'] = ab_stats.apply(
            lambda row: row['num_correct_R']/row['num_R'] if row['num_R'] > 0 else np.nan, axis=1)
        ab_stats['specificity'] = ab_stats.apply(
            lambda row: row['num_correct_S']/row['num_S'] if row['num_S'] > 0 else np.nan, axis=1)
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
    
    
    def _update_pheno_stats(self, batch_idx, pred_res: torch.Tensor, target_res: torch.Tensor, 
                          ab_mask: torch.Tensor, iso_stats_pheno: pd.DataFrame):
        indices = ab_mask.any(dim=1).nonzero().squeeze(-1).tolist() # list of isolates where phenotypes are present
        for idx in indices: 
            iso_ab_mask = ab_mask[idx]
            df_idx = batch_idx * self.val_batch_size + idx # index of the isolate in the combined dataset
            
            # counts
            num_masked_tot = iso_ab_mask.sum().item()
            num_masked_R = target_res[idx][iso_ab_mask].sum().item()
            num_masked_S = num_masked_tot - num_masked_R
            
            # statistics            
            iso_target_res = target_res[idx][iso_ab_mask]
            eq = torch.eq(pred_res[idx][iso_ab_mask], iso_target_res)
            num_correct_R = eq[iso_target_res == 1].sum().item()
            num_correct_S = eq[iso_target_res == 0].sum().item()
            num_correct = num_correct_S + num_correct_R
            all_correct = eq.all().item()
            
            data = {
                'num_masked': num_masked_tot, 'num_masked_S': num_masked_S, 'num_masked_R': num_masked_R, 
                'num_correct': num_correct, 'correct_S': num_correct_S, 'correct_R': num_correct_R,
                'all_correct': all_correct
            }
            iso_stats_pheno.loc[df_idx, data.keys()] = data.values()
                          
        return iso_stats_pheno
    
    
    def _update_geno_stats(self, batch_idx, token_pred: torch.Tensor, target_ids: torch.Tensor, 
                           token_mask: torch.Tensor, iso_stats_geno: pd.DataFrame):
        indices = token_mask.any(dim=1).nonzero().squeeze(-1).tolist() # list of isolates where genotypes are present
        for idx in indices:
            iso_token_mask = token_mask[idx]    
            df_idx = batch_idx * self.val_batch_size + idx # index of the isolate in the combined dataset
            
            num_masked = iso_token_mask.sum().item()
            pred_tokens = token_pred[idx, iso_token_mask].argmax(dim=-1)
            targets = target_ids[idx, iso_token_mask]

            eq = torch.eq(pred_tokens, targets)
            data = {
                'num_masked': num_masked, 'num_correct': eq.sum().item(), 'all_correct': eq.all().item()
            }
            iso_stats_geno.loc[df_idx, data.keys()] = data.values()
                
        return iso_stats_geno
    
    
    def _calculate_iso_stats(self, iso_stats_geno: pd.DataFrame, iso_stats_pheno: pd.DataFrame):
        
        iso_stats_geno['accuracy'] = iso_stats_geno['num_correct'] / iso_stats_geno['num_masked']
        
        iso_stats_pheno['accuracy'] = iso_stats_pheno['num_correct'] / iso_stats_pheno['num_masked']
        iso_stats_pheno['sensitivity'] = iso_stats_pheno.apply(
            lambda row: row['correct_R']/row['num_masked_R'] if row['num_masked_R'] > 0 else np.nan, axis=1
        )
        iso_stats_pheno['specificity'] = iso_stats_pheno.apply(
            lambda row: row['correct_S']/row['num_masked_S'] if row['num_masked_S'] > 0 else np.nan, axis=1
        )
        
        return iso_stats_geno, iso_stats_pheno
        
    
    def _init_wandb(self):
        self.wandb_run = wandb.init(
            project=self.project_name, # name of the project
            name=self.wandb_name, # name of the run
            
            config={
                "trainer_type": "pre-training",
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
                "weight_decay": self.weight_decay,
                "mask_prob_geno:": self.mask_prob_geno,
                "masking_method": self.train_set.masking_method,
                "mask_prob_pheno:": self.mask_prob_pheno if self.mask_prob_pheno else None,
                "num_known_ab": self.num_known_ab if self.num_known_ab else None,
                "num_known_classes": self.num_known_classes if self.num_known_classes else None,
                "max_seq_len": self.model.max_seq_len,
                "vocab_size": len(self.vocab),
                "num_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "num_antibiotics": self.num_ab,
                "antibiotics": self.antibiotics,
                "antibiotic weights:": self.ab_weights if self.wl_strength else None,
                "loss_fn": self.loss_fn,
                "train_size": self.train_size,
                "random_state": self.random_state,
                'val_share': self.val_share,
                "val_size": self.val_size,
            }
        )
        self.wandb_run.watch(self.model) # watch the model for gradients and parameters
        self.wandb_run.define_metric("epoch", hidden=True)
        self.wandb_run.define_metric("batch", hidden=True)
        
        self.wandb_run.define_metric("Losses/live_loss", step_metric="batch")
        self.wandb_run.define_metric("Losses/train_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Losses/geno_train_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Losses/pheno_train_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Losses/val_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Losses/val_geno_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Losses/val_pheno_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_pheno_acc", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_pheno_iso_acc", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_geno_acc", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_geno_iso_acc", summary="max", step_metric="epoch")
        
        self.wandb_run.define_metric("Losses/final_val_loss")
        self.wandb_run.define_metric("Losses/final_val_geno_loss")
        self.wandb_run.define_metric("Losses/final_val_pheno_loss")
        self.wandb_run.define_metric("Accuracies/final_val_pheno_acc")
        self.wandb_run.define_metric("Accuracies/final_val_pheno_iso_acc")
        self.wandb_run.define_metric("Accuracies/final_val_geno_acc")
        self.wandb_run.define_metric("Accuracies/final_val_geno_iso_acc")
        
        self.wandb_run.define_metric("best_epoch", hidden=True)
    
    
    def print_val_results(self, val_results: dict):
        print("Val loss: {:.4f} | Genotype loss: {:.4f} | Phenotype loss: {:.4f}".format(
            val_results['loss'], val_results['geno_loss'], val_results['pheno_loss']))
        print("Phenotype accuracy: {:.2%} | Phenotype isolate accuracy: {:.2%}".format(
            val_results['pheno_acc'], val_results['pheno_iso_acc']))
        print("Genotype accuracy: {:.2%} | Genotype isolate accuracy: {:.2%}".format(
            val_results['geno_acc'], val_results['geno_iso_acc']))
    
     
    def _report_epoch_results(self):
        if self.do_eval:
            wandb_dict = {
                "epoch": self.current_epoch+1,
                "Losses/train_loss": self.losses[-1],
                "Losses/val_loss": self.val_losses[-1],
                "Losses/val_geno_loss": self.val_geno_losses[-1],
                "Losses/val_pheno_loss": self.val_pheno_losses[-1],
                "Accuracies/val_pheno_acc": self.val_pheno_accs[-1],
                "Accuracies/val_pheno_iso_acc": self.val_pheno_iso_accs[-1],
                "Accuracies/val_geno_acc": self.val_geno_accs[-1],
                "Accuracies/val_geno_iso_acc": self.val_geno_iso_accs[-1],
            }
        else:
            wandb_dict = {
                "epoch": self.current_epoch+1,
                "Losses/train_loss": self.losses[-1],
                "Losses/geno_train_loss": self.geno_losses[-1],
                "Losses/pheno_train_loss": self.pheno_losses[-1],
                "Losses/val_loss": self.val_losses[-1],
                "Losses/val_geno_loss": self.val_geno_losses[-1],
                "Losses/val_pheno_loss": self.val_pheno_losses[-1]
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
        s += f" | Epoch: {self.current_epoch+1}/{self.epochs} | {batch_index}/{self.num_batches} ({progress:.2%}) | "\
                f"Loss: {mlm_loss:.4f}"
        print(s)
    
    
    def print_early_stop_results(self):
        print(f"Validation stats at best epoch ({self.best_epoch+1}):")
        s1 = f"Loss: {self.val_losses[self.best_epoch]:.4f}" 
        s1 += f"| Phenotype Loss: {self.val_pheno_losses[self.best_epoch]:.4f}"
        s1 += f" | Genotype Loss: {self.val_geno_losses[self.best_epoch]:.4f}"
        print(s1)
        s2 = f" | Phenotype accuracy: {self.val_pheno_accs[self.best_epoch]:.2%}"
        s2 += f" | Phenotype isolate accuracy: {self.val_pheno_iso_accs[self.best_epoch]:.2%}"
        print(s2)
        s3 = f" | Genotype accuracy: {self.val_geno_accs[self.best_epoch]:.2%}"
        s3 += f" | Genotype isolate accuracy: {self.val_geno_iso_accs[self.best_epoch]:.2%}"
        print(s3)
    
    
    def report_early_stop_results(self):
        self.wandb_run.log({
            "Losses/final_val_loss": self.best_val_loss, 
            "Losses/final_val_geno_loss": self.val_geno_losses[self.best_epoch],
            "Losses/final_val_pheno_loss": self.val_pheno_losses[self.best_epoch],
            "Accuracies/final_val_pheno_acc": self.val_pheno_accs[self.best_epoch],
            "Accuracies/final_val_pheno_iso_acc": self.val_pheno_iso_accs[self.best_epoch],
            "Accuracies/final_val_geno_acc": self.val_geno_accs[self.best_epoch],
            "Accuracies/final_val_geno_iso_acc": self.val_geno_iso_accs[self.best_epoch],
            "best_epoch": self.best_epoch+1
        })
    
    
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
        print("Model loaded")
        print("="*self._splitter_size)
        
