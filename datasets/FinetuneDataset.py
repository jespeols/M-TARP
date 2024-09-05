import numpy as np
import torch
import pandas as pd

from copy import deepcopy
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FinetuneDataset(Dataset):
    # df column names
    INDICES_MASKED = 'indices_masked' # input to BERT, token indices of the masked sequence
    TARGET_RESISTANCES = 'target_resistances' # resistance of the target antibiotics, what we want to predict
    TOKEN_TYPES = 'token_types' # # 0 for patient info, 1 for genotype, 2 for phenotype
    TARGET_IDS = 'target_ids' # indices of the target tokens for the genotype masking
    # if sequences are included
    MASKED_SEQUENCE = 'masked_sequence'
    
    def __init__(
        self,
        df_MM: pd.DataFrame, 
        vocab,
        antibiotics: list,
        specials: dict,
        max_seq_len: int,
        masking_method: str,
        mask_prob_geno: float,
        mask_prob_pheno: float,
        num_known_ab: int,
        num_known_classes: int = None,
        no_geno_masking: bool = False,
        always_mask_replace: bool = True,
        random_state: int = 42,
        include_sequences: bool = False,
    ):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state) # creates a new generator 
        
        self.ds = df_MM.reset_index(drop=True)
        assert all(self.ds['num_ab'] > 0), "Dataset contains isolates without phenotypes"
              
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics)
        self.ab_to_idx = {ab: idx for idx, ab in enumerate(self.antibiotics)}
        self.enc_res = {'S': 0, 'R': 1}
        self.max_seq_len = max_seq_len
        self.CLS, self.PAD = specials['CLS'], specials['PAD']
        self.AB_MASK, self.GENE_MASK = specials['AB_MASK'], specials['GENE_MASK']
        self.always_mask_replace = always_mask_replace
        if self.always_mask_replace:
            print("Always masking using MASK tokens")
        else:
            print("Masking using BERT 80-10-10 strategy")
        
        self.masking_method = masking_method # 'random', 'num_known_ab' or 'num_known_classes' 
        self.mask_prob_geno = mask_prob_geno
        self.no_genotype_masking = no_geno_masking
        self.mask_prob_pheno = mask_prob_pheno
        self.num_known_ab = num_known_ab
        self.num_known_classes = num_known_classes
        if self.masking_method == 'random':
            assert self.mask_prob_pheno, "mask_prob_pheno must be given if masking_method is 'random'"
        elif self.masking_method == 'num_known_ab':
            assert self.num_known_ab, "num_known_ab must be given if masking_method is 'num_known'"
            self.ds = self.ds[self.ds['num_ab'] > self.num_known_ab].reset_index(drop=True)
        elif self.masking_method == 'num_known_classes':
            assert num_known_classes, "num_known_classes must be given if masking_method is 'num_known_classes'"
            self.ds = self.ds[self.ds['ab_classes'].apply(lambda x: len(set(x)) > self.num_known_classes)].reset_index(drop=True)
        self.num_samples = self.ds.shape[0]
                    
        self.include_sequences = include_sequences
        self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES, self.TARGET_IDS, self.TOKEN_TYPES]
        if self.include_sequences:
            self.columns += [self.MASKED_SEQUENCE]
            
    
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        target_res = torch.tensor(item[self.TARGET_RESISTANCES], dtype=torch.float32, device=device)
        token_types = torch.tensor(item[self.TOKEN_TYPES], dtype=torch.long, device=device)
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        target_ids = torch.tensor(item[self.TARGET_IDS], dtype=torch.long, device=device)
        
        if self.include_sequences:
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_res, target_ids, token_types, attn_mask, masked_sequence
        else:
            return input, target_res, target_ids, token_types, attn_mask   
    
    
    def prepare_dataset(self):
        geno_sequences = deepcopy(self.ds['genotypes'].tolist())
        pheno_sequences = deepcopy(self.ds['phenotypes'].tolist())
        years = self.ds['year'].astype(str).tolist()
        countries = self.ds['country'].tolist()
        seq_starts = [[self.CLS, years[i], countries[i]] for i in range(self.num_samples)]
        
        if self.masking_method == "num_known_classes":
            ab_classes = deepcopy(self.ds['ab_classes'].tolist())
            masked_pheno_sequences, target_resistances, target_ids_pheno = self._mask_pheno_sequences(pheno_sequences, ab_classes)
        else:
            masked_pheno_sequences, target_resistances, target_ids_pheno = self._mask_pheno_sequences(pheno_sequences)
            
        pheno_token_types = [[2]*len(seq) for seq in masked_pheno_sequences]
        if self.no_genotype_masking:
            masked_geno_sequences = geno_sequences
            target_ids_geno = [[-1]*len(seq) for seq in masked_geno_sequences]
        else:
            masked_geno_sequences, target_ids_geno = self._mask_geno_sequences(geno_sequences)
        geno_token_types = [[1]*len(seq) for seq in masked_geno_sequences]
        
        # combine sequences and pad
        target_ids = [[-1]*3 + target_ids_geno[i] + target_ids_pheno[i] for i in range(self.num_samples)]
        target_ids = [indices + [-1]*(self.max_seq_len - len(indices)) for indices in target_ids]
        
        masked_sequences = [seq_starts[i] + masked_geno_sequences[i] + masked_pheno_sequences[i] for i in range(self.num_samples)]
        masked_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_sequences]
        indices_masked = [self.vocab.lookup_indices(seq) for seq in masked_sequences]
        
        token_types = [[0]*3 + geno_token_types[i] + pheno_token_types[i] for i in range(self.num_samples)]
        token_types = [seq + [2]*(self.max_seq_len - len(seq)) for seq in token_types]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_resistances, target_ids, token_types, masked_sequences)
        else:
            rows = zip(indices_masked, target_resistances, target_ids, token_types)
        self.df = pd.DataFrame(rows, columns=self.columns)
         
         
    def _mask_geno_sequences(self, geno_sequences): # Just to remove info, no prediction task
        masked_geno_sequences = list()
        geno_target_ids = list() 
        
        for geno_seq in geno_sequences:
            seq_len = len(geno_seq)
            target_ids = np.array([-1]*seq_len)
            if not seq_len == 0:
                token_mask = self.rng.random(seq_len) < self.mask_prob_geno
                if not token_mask.any():
                    token_mask[self.rng.integers(seq_len)] = True
                masking_indices = token_mask.nonzero()[0]
                target_ids[token_mask] = self.vocab.lookup_indices([geno_seq[i] for i in masking_indices])
                for idx in masking_indices:
                    geno_seq[idx] = self.GENE_MASK
            masked_geno_sequences.append(geno_seq)
            geno_target_ids.append(target_ids.tolist())
        return masked_geno_sequences, geno_target_ids
    
    
    def _get_replace_token(self, mask_token, original_token): 
        if self.always_mask_replace:
            return mask_token
        else:                       ## BERT masking
            r = self.rng.random()
            if r < 0.8:
                return mask_token
            elif r < 0.9:
                return self.vocab.lookup_token(self.rng.integers(self.vocab_size))
            else:
                return original_token
    
    
    def _mask_pheno_sequences(self, pheno_sequences, ab_classes=None):
        masked_pheno_sequences = list()
        target_resistances = list()
        pheno_target_ids = list()

        if self.masking_method == 'random':
            for pheno_seq in pheno_sequences:
                seq_len = len(pheno_seq)
                token_mask = self.rng.random(seq_len) < self.mask_prob_pheno
                target_res = [-1]*self.num_ab
                target_ids = np.array([-1]*seq_len)
                if not token_mask.any():
                    token_mask[self.rng.integers(seq_len)] = True 
                masking_indices = token_mask.nonzero()[0]
                target_ids[token_mask] = self.vocab.lookup_indices([pheno_seq[i] for i in masking_indices])
                for idx in masking_indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())
        elif self.masking_method == 'num_known_ab':
            for pheno_seq in pheno_sequences:
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                target_ids = np.array([-1]*seq_len)  
                masking_indices = self.rng.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                target_ids[masking_indices] = self.vocab.lookup_indices([pheno_seq[i] for i in masking_indices])
                for idx in masking_indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])                
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())
        elif self.masking_method == 'num_known_classes':
            for i, pheno_seq in enumerate(pheno_sequences):
                classes = ab_classes[i]     
                ## randomly choose one class to keep        
                unique_classes = np.unique(classes)
                keep_classes = self.rng.choice(unique_classes, self.num_known_classes, replace=False) # all classes are equally likely
                seq_len = len(pheno_seq)
                target_ids = np.array([-1]*seq_len)
                target_res = [-1]*self.num_ab
                indices = [idx for idx in range(seq_len) if classes[idx] not in keep_classes] 
                target_ids[indices] = self.vocab.lookup_indices([pheno_seq[i] for i in indices])
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")
        
        return masked_pheno_sequences, target_resistances, pheno_target_ids