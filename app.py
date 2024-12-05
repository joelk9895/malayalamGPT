import torch
import json
import torch.nn as nn
from torch.nn import functional as F
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
from fastapi.middleware.cors import CORSMiddleware

with open('tokenization.json', 'r') as f:
    tokenization = json.load(f)
stoi = tokenization['stoi']
itos = tokenization['itos']
vocab_size = len(stoi)

# Model definition (same as provided)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Hyperparameters and Model Setup
n_embd = 384
n_head = 6
n_layer = 6
block_size = 256
dropout = 0.2
device = 'cpu' if not torch.cuda.is_available() else 'cuda'

model = GPTLanguageModel()
model = model.to(device)
checkpoint = torch.load('malayalam_gpt_checkpoint_40000.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")

# FastAPI Server
app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5500", "https://hksw4q5b-5500.inc1.devtunnels.ms/"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
@app.get("/generate")
async def stream_response(context: str):
    # Tokenize the input context
    context_encoded = [stoi[c] for c in context if c in stoi]
    
    # Handle empty or invalid input
    if not context_encoded:
        return StreamingResponse(
            iter(["data: Invalid input context\n\n"]), media_type="text/event-stream"
        )

    # Convert to tensor and initialize
    context_tensor = torch.tensor(context_encoded, dtype=torch.long, device=device).unsqueeze(0)

    async def event_stream():
        try:
            generated = context_tensor
            output_text = ""
            for _ in range(500):  # Generate up to 500 tokens
                generated = model.generate(generated, max_new_tokens=1)
                output = ''.join([itos[str(idx)] for idx in generated[0].tolist()])
                chunk = output[-1]
                output_text += chunk
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.01)  # Simulate streaming delay
                if len(output_text) >= 500:
                    break
        except Exception as e:
            yield f"data: Error occurred: {str(e)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")