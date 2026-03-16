# Building & Training Modern LLM - Detailed Notes

This document explains the key components of building a modern Large Language Model (LLM) from scratch. The notebook implements 4 major improvements over the basic transformer: **RMSNorm**, **RoPE**, **GQA**, and **SwiGLU**.

---

## Table of Contents
1. [Setup & Data Preparation](#section-0-setup--data-preparation)
2. [RMSNorm (Root Mean Square Layer Normalization)](#section-1-rmsnorm)
3. [RoPE (Rotary Positional Embeddings)](#section-2-rope-rotary-positional-embeddings)
4. [Grouped Query Attention (GQA)](#section-3-grouped-query-attention-gqa)
5. [SwiGLU Feed-Forward Network](#section-4-swiglu-feed-forward-network)
6. [Full Model Assembly](#section-5-assemble-the-full-model)
7. [Training Loop](#section-6-training)
8. [Text Generation](#section-7-generation)

---

## Section 0: Setup & Data Preparation

### What is Tokenization?
Before feeding text to a neural network, we need to convert characters into numbers. This notebook uses **character-level tokenization**:

```python
chars = sorted(set(text))  # Find all unique characters
vocab_size = len(chars)    # Count them
```

- Shakespeare text has 65 unique characters (letters, punctuation, spaces)
- Each character gets a unique number (0-64)
- `char_to_idx` dictionary maps character → number
- `idx_to_char` dictionary maps number → character

### Why Character Level?
Simple but inefficient. Real LLMs use BPE (Byte Pair Encoding) or WordPiece tokenization which groups characters into meaningful chunks like "ing", "tion", etc.

### Train/Val Split
```python
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]   # 90% for training
val_data = data[n:]     # 10% for validation
```

We split data to check if the model generalizes well (validation loss) or just memorizes training data (overfitting).

### Batching - Getting Random Chunks
```python
def get_batch(split, batch_size, context_length):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - context_length, (batch_size,))  # Random starting positions
    x = torch.stack([d[i:i+context_length] for i in ix])       # Input sequence
    y = torch.stack([d[i+1:i+context_length+1] for i in ix])   # Target (shifted by 1)
    return x.to(device), y.to(device)
```

**Key concept**: Language models predict the next token. If input is "hello", target is "ello" (shifted by 1).

Example:
- Input: `"s his re"` 
- Target: `" his rea"` (each character predicts the next one)

---

## Section 1: RMSNorm

### What is Normalization?
In neural networks, values can get too large or too small. **Normalization** keeps numbers in a healthy range so the network trains well.

### LayerNorm (What You Might Know)
```python
# Pseudocode
x = x - x.mean()           # Subtract mean (center at 0)
x = x / x.std()            # Divide by standard deviation (scale to 1)
x = x * weight + bias      # Learnable scale and shift
```

### RMSNorm - Simpler & Faster

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale only

    def forward(self, x):
        # Step-by-step:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

### Line-by-Line Explanation:

```python
rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
```

| Part | Explanation |
|------|-------------|
| `x.pow(2)` | Square every number (x²) - makes all values positive |
| `.mean(dim=-1)` | Average across the last dimension (features). `dim=-1` means the last dimension |
| `keepdim=True` | Keeps the dimension so we can divide properly (shape doesn't collapse) |
| `+ self.eps` | Small value (1e-6) to prevent division by zero |
| `torch.sqrt()` | Square root - this gives us **Root Mean Square (RMS)** |

```python
return (x / rms) * self.weight
```

- `x / rms` - Divide input by its RMS (normalizes to unit RMS)
- `* self.weight` - Multiply by learnable scale (allows network to adjust)

### Why No Mean Subtraction?
Research showed that mean centering doesn't help much. By removing it:
- Fewer operations = faster
- Fewer parameters (no bias term)
- Works just as well in practice

---

## Section 2: RoPE (Rotary Positional Embeddings)

### The Problem with Position
Transformers process all tokens at once (parallel), so they don't naturally know the order. We need to tell them "this token is at position 3, that one at position 10".

### Old Approach: Sinusoidal Positional Encoding
Add fixed sine/cosine patterns to embeddings. Works but:
- Cannot handle long sequences well
- Absolute position only (doesn't capture relative distance)

### RoPE Solution: Rotate the Vectors!

```python
def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    """
    Create rotation angles for each position and dimension pair.
    """
    # 1. Create frequency values for each dimension pair
    # Example: head_dim=8, we get [0, 2, 4, 6] -> divided by 8 -> [0, 0.25, 0.5, 0.75]
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # 2. Create position values [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len).float()
    
    # 3. Outer product: each position × each frequency = angle
    angles = torch.outer(positions, freqs)  # [max_seq_len, head_dim // 2]
    
    # 4. Return cosine and sine of angles
    return torch.cos(angles), torch.sin(angles)
```

**Intuition**: 
- Low dimensions (first rows) → high frequency → captures short-range patterns
- High dimensions (last rows) → low frequency → captures long-range patterns

### Applying RoPE - The Rotation Math

```python
def apply_rope(x, cos, sin):
    """
    Apply rotary embeddings to Q and K tensors.
    x shape: [batch, n_heads, seq_len, head_dim]
    """
    seq_len = x.shape[2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, hd//2]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    
    # Split into even and odd dimensions
    x1 = x[..., ::2]   # dimensions 0, 2, 4, 6, ...
    x2 = x[..., 1::3]  # dimensions 1, 3, 5, 7, ...
    
    # Rotation formula (2D rotation matrix):
    # [cos -sin] [x1] = [x1*cos - x2*sin]
    # [sin  cos] [x2]   [x1*sin + x2*cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    
    # Interleave back together
    return torch.stack([out1, out2], dim=-1).flatten(-2)
```

### Why RoPE Captures Relative Position
When you compute attention (dot product of Q and K), the rotation angles cancel out in a way that only depends on the **difference** in positions, not absolute positions!

Position 3 vs Position 5: difference = 2
Position 7 vs Position 9: difference = 2

The model sees these as the same! This is exactly what we want for understanding language.

---

## Section 3: Grouped Query Attention (GQA)

### Standard Multi-Head Attention (MHA)
```python
# In standard attention:
self.q_proj = nn.Linear(d_model, n_heads * head_dim)  # Each head gets Q
self.k_proj = nn.Linear(d_model, n_heads * head_dim)  # Each head gets K
self.v_proj = nn.Linear(d_model, n_heads * head_dim)  # Each head gets V
```

**Problem**: If we have 32 query heads, we need 32 K heads and 32 V heads. That's a lot of memory, especially for long sequences (KV cache gets huge!).

### GQA - Share K and V Among Groups

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        # n_heads = 8 (query heads)
        # n_kv_heads = 2 (key/value heads - MUCH fewer!)
        # n_rep = 8 // 2 = 4 (each KV head is shared by 4 Q heads)
        
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
```

### The repeat_kv Function

```python
def repeat_kv(x, n_rep):
    """Repeat KV heads to match Q heads."""
    if n_rep == 1:
        return x
    
    b, n_kv, seq, hd = x.shape  # batch, kv_heads, seq_len, head_dim
    # Expand: [b, n_kv, 1, seq, hd] -> [b, n_kv, n_rep, seq, hd] -> reshape
    return (x[:, :, None, :, :]
            .expand(b, n_kv, n_rep, seq, hd)
            .reshape(b, n_kv * n_rep, seq, hd))
```

**Visual**:
```
With 8 Q heads and 2 KV heads (n_rep=4):

Q heads:    [Q0] [Q1] [Q2] [Q3] [Q4] [Q5] [Q6] [Q7]
                 ↓     ↓     ↓     ↓
KV heads:      [K0,V0]        [K1,V1]
                 ↓     ↓     ↓     ↓
Repeated:   [K0][K0][K0][K0] [K1][K1][K1][K1]
```

### Attention Computation

```python
def forward(self, x, rope_cos, rope_sin):
    b, seq, _ = x.shape
    
    # 1. Project to Q, K, V
    q = self.q_proj(x).view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
    
    # 2. Apply RoPE to Q and K (not V - values don't need position!)
    q = apply_rope(q, rope_cos, rope_sin)
    k = apply_rope(k, rope_cos, rope_sin)
    
    # 3. Repeat KV heads to match Q heads
    k = repeat_kv(k, self.n_rep)
    v = repeat_kv(v, self.n_rep)
    
    # 4. Scaled dot-product attention
    scale = 1.0 / math.sqrt(self.head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale  # Matrix multiply
    
    # 5. Causal mask (upper triangle = -inf) - prevents seeing future
    mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    
    # 6. Softmax to get attention weights
    weights = F.softmax(scores, dim=-1)
    
    # 7. Dropout for regularization
    weights = F.dropout(weights, p=DROPOUT, training=self.training)
    
    # 8. Apply attention to values
    out = weights @ v
    
    # 9. Merge heads and project back
    out = out.transpose(1, 2).contiguous().view(b, seq, -1)
    return self.o_proj(out)
```

### Why GQA is Great
- **4x smaller KV cache** with 8 Q heads and 2 KV heads
- Similar quality to full MHA
- Critical for long context models (Llama 2, Mistral use this!)

---

## Section 4: SwiGLU Feed-Forward Network

### Standard FFN
```python
# Old approach:
self.fc1 = nn.Linear(d_model, hidden_dim)  # Expand: 256 -> 1024
self.fc2 = nn.Linear(hidden_dim, d_model)  # Contract: 1024 -> 256
# Activation in between: ReLU
```

### SwiGLU - Gated Linear Unit

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)  # Gate path
        self.w_up   = nn.Linear(d_model, hidden_dim, bias=False)  # Value path
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)  # Output
    
    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # SiLU activation (also called Swish)
        up   = self.w_up(x)            # Value/path information
        return F.dropout(self.w_down(gate * up), p=DROPOUT, training=self.training)
```

### Understanding SiLU
```python
# SiLU(x) = x * sigmoid(x)
# Also called Swish: f(x) = x * sigmoid(βx) where β=1 for SiLU
```

| Activation | Formula | Properties |
|------------|---------|------------|
| ReLU | max(0, x) | Hard cutoff at 0, not smooth |
| SiLU | x * sigmoid(x) | Smooth, slight negative values for x<0 |

### Why Gating Works
```
gate = SiLU(W_gate × x)  # Decides HOW MUCH information passes
up   = W_up × x          # Carries the INFORMATION

output = gate × up       # Element-wise multiplication
        = SiLU(W_gate × x) ⊙ (W_up × x)
```

The gate can learn to:
- Let certain dimensions pass through strongly (gate ≈ 1)
- Block certain dimensions (gate ≈ 0)
- This is adaptive - the network decides what to keep!

---

## Section 5: Assemble the Full Model

### Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, ffn_hidden_dim):
        self.attn_norm = RMSNorm(d_model)           # Pre-norm for attention
        self.attention = GroupedQueryAttention(...) # Self-attention
        self.ffn_norm = RMSNorm(d_model)            # Pre-norm for FFN
        self.ffn = SwiGLU(...)                      # Feed-forward network
    
    def forward(self, x, rope_cos, rope_sin):
        # Pre-norm architecture (better training stability)
        x = x + self.attention(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

**Key**: Notice the `+` (residual connections). This helps gradient flow and prevents vanishing gradients.

### Full MiniLLM Model

```python
class MiniLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_hidden_dim, max_seq_len):
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_kv_heads, ffn_hidden_dim)
            for _ in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: embedding and output layer share weights!
        self.lm_head.weight = self.token_emb.weight
        
        # Precompute RoPE frequencies
        head_dim = d_model // n_heads
        rope_cos, rope_sin = precompute_rope_freqs(head_dim, max_seq_len)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
        
        self.apply(self._init_weights)
    
    def forward(self, idx, targets=None):
        x = self.token_emb(idx)  # Convert token IDs to embeddings
        
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)  # Project to vocabulary size
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
```

### Weight Tying
```python
self.lm_head.weight = self.token_emb.weight
```

This means the matrix that converts embeddings→vocabulary is the same as vocabulary→embeddings. Saves parameters and often improves performance!

---

## Section 6: Training

### The Training Loop

```python
for step in range(MAX_STEPS):
    # 1. Get a batch of data
    xb, yb = get_batch("train", BATCH_SIZE, CONTEXT_LEN)
    
    # 2. Forward pass: get predictions and loss
    logits, loss = model(xb, yb)
    
    # 3. Clear previous gradients
    optimizer.zero_grad()
    
    # 4. Backward pass: compute gradients
    loss.backward()
    
    # 5. Gradient clipping: prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 6. Update weights
    optimizer.step()
```

### Cross Entropy Loss Explained
```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

- `logits`: Raw scores for each character (65 values per position)
- `targets`: The actual next character (0-64)
- Cross entropy measures how "surprised" the model is by the actual next character
- Lower loss = better predictions

### Why Gradient Clipping?
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

If gradients get too huge, training becomes unstable. This clips them to max norm of 1.0.

### Evaluation
```python
@torch.no_grad()
def estimate_loss():
    model.eval()  # Turn off dropout
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(EVAL_STEPS):
            xb, yb = get_batch(split, BATCH_SIZE, CONTEXT_LEN)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()  # Turn dropout back on
    return out
```

- `@torch.no_grad()` - Don't compute gradients (faster, less memory)
- `model.eval()` / `model.train()` - Switch between training and evaluation modes

---

## Section 7: Generation (Autoregressive)

### How Text Generation Works

```python
@torch.no_grad()
def generate(model, prompt, max_new_tokens=500, temperature=0.8):
    model.eval()
    tokens = encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        # Keep only last max_seq_len tokens (memory efficiency)
        context = tokens[:, -config["max_seq_len"]:]
        
        # Get predictions
        logits, _ = model(context)
        
        # Get last position's logits only
        logits = logits[:, -1, :] / temperature
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample next token (can be random!)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return decode(tokens[0].tolist())
```

### Temperature Explained
```python
logits = logits / temperature
```

| Temperature | Effect |
|-------------|--------|
| Low (0.1-0.3) | Greedy - always picks most likely. Repetitive, safe |
| Medium (0.7-0.9) | Balanced - variety but mostly sensible |
| High (1.0+) | Random - creative but can be nonsense |

**Low temperature**: 
- `logits / 0.1` → very large values → softmax → almost deterministic

**High temperature**:
- `logits / 2.0` → small values → softmax → more uniform distribution

### Why Sample Instead of Greedy?
Greedy (always pick highest probability) leads to repetitive, boring text. Sampling adds creativity!

---

## Model Configuration Summary

```python
config = {
    "vocab_size": 65,           # Characters
    "d_model": 256,             # Embedding dimension
    "n_layers": 4,              # Number of transformer blocks
    "n_heads": 8,               # Query heads
    "n_kv_heads": 2,            # Key/Value heads (GQA)
    "ffn_hidden_dim": 680,      # FFN hidden dimension (≈ 2.67 * d_model)
    "max_seq_len": 256,         # Context length
}
```

**Total parameters**: ~2.7 million (tiny! Real LLMs have billions)

---

## Key Takeaways

1. **RMSNorm**: Faster than LayerNorm, no mean subtraction, no bias
2. **RoPE**: Rotates Q and K to encode relative position - no learned positional embeddings!
3. **GQA**: Fewer KV heads than Q heads - saves memory, same quality
4. **SwiGLU**: Gated activation - network learns which info to keep
5. **Pre-norm**: Apply normalization before attention/FFN (more stable training)
6. **Weight tying**: Share embedding and output weights
7. **Autoregressive generation**: Predict one token at a time, feed back into model

---

## Further Experiments to Try

1. **Temperature sweep**: Try T=0.1, 0.5, 1.0, 1.5, 2.0
2. **Model size**: Double d_model to 512 - what happens to training?
3. **Layers**: Add more transformer layers
4. **GQA ratio**: Try n_heads=8, n_kv_heads=4 (2:1 ratio instead of 4:1)
5. **Data**: Train on different text (Python code, Hindi, etc.)

---

*Notes generated from: buildng_and_training_modern_LLM.ipynb*