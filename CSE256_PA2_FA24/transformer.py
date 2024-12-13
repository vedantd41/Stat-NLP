import torch
from torch import nn
from torch.nn import functional as F

# MultiHeadAttention class for handling multiple attention heads in a transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, is_decoder, n_embd, num_heads, head_size, block_size, dropout_rate=0.3):
        super().__init__()
        
        head_class = AttentionHead
        # Initialize attention heads
        self.heads = nn.ModuleList([head_class(is_decoder=is_decoder, 
                                              head_size=head_size, 
                                              embedding_size=n_embd, 
                                              block_size=block_size, 
                                              dropout_rate=dropout_rate) 
                                   for _ in range(num_heads)])
        # Projection layer to combine outputs from multiple heads
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out_list, attn_maps = [], []

        # Process each attention head      
        for head in self.heads:
            out, attn_map = head(x)
            out_list.append(out)
            attn_maps.append(attn_map)
       
        # Concatenate the outputs from all heads and apply dropout after projection
        out = torch.cat(out_list, dim=-1)
        out = self.dropout(self.proj(out))
        
        return out, attn_maps

class Block(nn.Module):
    def __init__(self, is_decoder, n_embd, n_head, block_size, dropout_rate=0.3):
        super().__init__()
        
        head_size = n_embd // n_head
        # Self-attention layer
        self.self_attention = MultiHeadAttention(
            is_decoder=is_decoder, 
            n_embd=n_embd, 
            num_heads=n_head, 
            head_size=head_size, 
            dropout_rate=dropout_rate, 
            block_size=block_size
        )
        # Feedforward network after attention
        self.feedforward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        att_out, attention_maps = self.self_attention(self.ln1(x))
        x = x + att_out # Residual connection after attention
        x = x + self.feedforward(self.ln2(x)) # Residual connection after feedforward
        return x, attention_maps 

# Attention Head class (calculates attention weights and applies them)
class AttentionHead(nn.Module):
    def __init__(self, head_size, embedding_size, block_size, dropout_rate=0.3, is_decoder=False):
        super().__init__()
        # Linear layers for query, key, and value projections
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        # Lower triangular matrix for masking (if decoder)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)
        self.is_decoder = is_decoder

    def forward(self, x):
        b, t, c = x.shape
        # Compute key, query, and value
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # Compute attention weights (scaled dot product)
        wei = q @ k.transpose(-2, -1) * c**-0.5  # (B, T, T)
        # Apply causal mask for decoder (future tokens are masked)
        if self.is_decoder:
            wei = wei.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # Apply attention to value (weighted sum)
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
        
        return out, wei

# Feedforward network for each block (fully connected layers)
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

# Encoder class (stack of transformer blocks)
class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, is_decoder, use_alibi=False, use_cls_token=False, dropout_rate=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_alibi = use_alibi
        self.use_cls_token = use_cls_token
        self.block_size = block_size
        self.word_embd_table = nn.Embedding(self.vocab_size, n_embd)
        if use_cls_token:
            self.vocab_size += 1 
            self.block_size += 1
        if not use_alibi:
            self.pos_embd_table = nn.Embedding(self.block_size, n_embd)
        
        self.blocks = nn.ModuleList([Block(is_decoder=is_decoder, n_embd=n_embd, n_head=n_head, block_size=self.block_size, dropout_rate=dropout_rate) for _ in range(n_layer)])

    def forward(self, x):
        b, t = x.shape
        
        # If using CLS token, insert it at the start of the sequence
        if self.use_cls_token:
            # cls_token = torch.zeros(b, 1, device=x.device).long()  # Batch size x 1 CLS token
            # x = torch.cat([cls_token, x], dim=1)  # Add CLS token to the input sequence
            t += 1  # Increase the length of the sequence
            cls_token_index = 0 # Define CLS token index
            cls_token = torch.full((b, 1), cls_token_index, device=x.device).long()  # Batch size x 1 CLS token
            x = torch.cat([cls_token, x], dim=1)
        
        word_embd = self.word_embd_table(x)
        
        if self.use_alibi:
            pos_embd = self.generate_alibi(t, word_embd.size(-1), x.device)
        else:
            pos_embd = self.pos_embd_table(torch.arange(t, device=x.device))
            
        x = word_embd + pos_embd
        attention_maps = []
        
        for block in self.blocks:
            x, block_attention = block(x)
            attention_maps.append(block_attention)
        
        return x, attention_maps

    def generate_alibi(self, seq_len, dim, device):
        alibi = torch.arange(seq_len, device=device).unsqueeze(1).repeat(1, dim) * (-0.05)
        return alibi

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

class EncoderClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        out_embds, attn_maps = self.encoder(x)
        
        # If using CLS token, take the embedding of the CLS token for classification
        if self.encoder.use_cls_token:
            cls_embd = out_embds[:, 0]
        else:
            cls_embd = out_embds.mean(dim=1) 
        
        out = self.classifier(cls_embd)
        return out, attn_maps

# Decoder class for language modelling
class Decoder(nn.Module):
    def __init__(self, is_decoder, vocab_size, n_embd, block_size, n_head, n_layer, use_alibi=False):
        super().__init__()
        
        self.use_alibi = use_alibi
        self.word_embd_table = nn.Embedding(vocab_size, n_embd)
        
        if not use_alibi:
            self.pos_embd_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            Block(is_decoder, 
                  n_embd, 
                  n_head=n_head, 
                  block_size=block_size,
                  ) 
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        
        word_embd = self.word_embd_table(idx)
        
        # Use AliBi 
        if self.use_alibi:
            pos_embd = self.generate_alibi(t, word_embd.size(-1), idx.device)
        else:
            pos_embd = self.pos_embd_table(torch.arange(t, device=idx.device))
        
        x = word_embd + pos_embd
                
        attention_maps = []
        
        for block in self.blocks:
            x, block_attention = block(x)
            attention_maps.append(block_attention)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        probs = [] 
        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = self.loss(logits, targets)
            probs = self.softmax(x)

        return logits, loss, attention_maps, probs

    def generate_alibi(self, seq_len, dim, device):
        alibi = torch.arange(seq_len, device=device).unsqueeze(1).repeat(1, dim) * (-0.05)
        return alibi