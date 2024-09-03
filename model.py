# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JointEmbedding(nn.Module):
    
    def __init__(self, config: dict, vocab_size: int, max_seq_len: int, pad_idx):
        super(JointEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.emb_dim = config['emb_dim']
        self.dropout_prob = config['dropout_prob']
        
        self.token_emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.pad_idx) 
        self.token_type_emb = nn.Embedding(3, self.emb_dim) # 3 possible values for token type: 0, 1, 2
        
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
    def forward(self, input_tensor: torch.Tensor, token_type_tensor: torch.Tensor):
        
        token_emb = self.token_emb(input_tensor) # (batch_size, seq_len, emb_dim)
        token_type_emb = self.token_type_emb(token_type_tensor) # (batch_size, seq_len, emb_dim)
        
        emb = token_emb + token_type_emb
        emb = self.layer_norm(emb) 
        emb = self.dropout(emb)
        return emb
    
################################################################################################################

class MultiHeadAttention(nn.Module):
    
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        
        self.emb_dim = config['emb_dim']
        self.num_heads = config['num_heads']
        self.dropout_prob = config['dropout_prob']
        
        self.head_dim = self.emb_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.emb_dim, f"Embedding dimension must be divisible by number of heads, got {self.emb_dim} and {self.num_heads}"
        
        self.q = nn.Linear(self.emb_dim, self.emb_dim)
        self.k = nn.Linear(self.emb_dim, self.emb_dim)
        self.v = nn.Linear(self.emb_dim, self.emb_dim)
        
        self.dropout = nn.Dropout(self.dropout_prob)
    
    def forward(self, input_emb: torch.Tensor, attn_mask:torch.Tensor = None):
        B, L, D = input_emb.size() # (B=batch_size, L=seq_len, D=emb_dim)
        
        # project input embeddings to query, key, value, then split into num_heads, reducing the embedding dimension
        query = self.q(input_emb).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # (B, num_heads, L, head_dim)
        key = self.k(input_emb).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # (B, num_heads, L, head_dim)
        value = self.v(input_emb).view(B, L, self.num_heads, self.head_dim).transpose(1,2) # (B, num_heads, L, head_dim)
        
        scale_factor = query.size(-1) ** 0.5
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) / scale_factor # (B, num_heads, L, L)
        
        attn_scores = attn_scores.masked_fill_(~attn_mask, -1e9) if attn_mask is not None else attn_scores 
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, num_heads, L, L)
        attn_weights = self.dropout(attn_weights)
        
        attn = torch.matmul(attn_weights, value) # (B, num_heads, L, head_dim)
        attn = attn.transpose(1, 2).contiguous().view(B, L, D) # (B, L, num_heads, head_dim) -> (B, L, D), concatenate the head
        return attn
        

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        
        self.emb_dim = config['emb_dim']
        self.ff_dim = config['ff_dim']
        self.num_heads = config['num_heads']
        self.dropout_prob = config['dropout_prob']
        
        self.attention = MultiHeadAttention(config)
        
        self.dense = nn.Linear(self.emb_dim, self.emb_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.emb_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(self.emb_dim, self.ff_dim), # ff_dim is the intermediate dimension, need not be = emb_dim
            nn.GELU(),
            nn.Linear(self.ff_dim, self.emb_dim),
            nn.Dropout(self.dropout_prob)
        )
        
    def forward(self, input_emb: torch.Tensor, attn_mask: torch.Tensor):
        x = input_emb
        attn = self.attention(x, attn_mask)
        
        hidden_states = self.dense(attn)
        hidden_states = self.dropout(hidden_states)
        x = x + hidden_states
        x = self.layer_norm(x)
        
        res = x
        x = self.feed_forward(x)
        x = x + res
        x = self.layer_norm(x)
        return x
    
    
class Model(nn.Module):
    
    def __init__(
            self, 
            config,
            vocab_size: int,
            max_seq_len: int,
            num_ab: int,
            pad_idx: int = 1,
            pheno_only: bool = False
        ):
        super(Model, self).__init__()
        
        self.random_state = config['random_state']
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        
        self.vocab_size = vocab_size
        self.num_ab = num_ab
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.pheno_only = pheno_only # if True, only the antibiotic resistance classification task is performed
        
        self.is_pretrained = False
                
        # parameters
        self.emb_dim = config['emb_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.ff_dim = config['ff_dim']
        self.dropout_prob = config['dropout_prob']
        
        # embedding and encoder blocks
        self.embedding = JointEmbedding(config, self.vocab_size, self.max_seq_len, self.pad_idx) 
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(self.num_layers)])
        
        self.token_prediction_layer = nn.Linear(self.emb_dim, self.vocab_size) # MLM task
        
        self.hidden_dim = config['hidden_dim'] # for the classification layer
        self.classification_layer = nn.ModuleList([AbPredictor(self.emb_dim, self.hidden_dim) for _ in range(num_ab)])
        
    def forward(self, input_tensor: torch.Tensor, token_type_tensor: torch.Tensor, attn_mask:torch.Tensor): 
        embedded = self.embedding(input_tensor, token_type_tensor)
        for layer in self.encoder:
            embedded = layer(embedded, attn_mask)
        encoded = embedded # ouput of the BERT Encoder
        
        cls_token = encoded[:, 0, :] # (batch_size, emb_dim)
        resistance_predictions = torch.cat([net(cls_token) for net in self.classification_layer], dim=1) # (batch_size, num_ab)
        if not self.pheno_only:
            token_predictions = self.token_prediction_layer(encoded) # (batch_size, seq_len, vocab_size)
            return resistance_predictions, token_predictions
        else:
            return resistance_predictions


class AbPredictor(nn.Module): # predicts resistance or susceptibility for an antibiotic
    def __init__(self, emb_dim: int, hidden_dim: int):
        super(AbPredictor, self).__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1), # binary classification (S:0 | R:1)
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
           
    def forward(self, X):
        # X is the CLS token of the BERT model
        return self.classifier(X)