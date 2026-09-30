import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):
        return self.embedding(token_ids)

class PositionalEmbedding(nn.Module):
    def __init__(self, block_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(block_size, d_model)

    def forward(self, token_embeddings):
        batch_size, sequence_length, _ = token_embeddings.shape

        positions = torch.arange(sequence_length, device=token_embeddings.device)
        position_embeddings = self.embedding(positions)

        return position_embeddings

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, d_model):
        super().__init__()

        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_embedding = PositionalEmbedding(block_size, d_model)

    def forward(self, token_ids):
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(token_embeddings)

        return token_embeddings + position_embeddings

class SelfAttention(nn.Module):
    def __init__(self, d_model, head_dim): #head_dim is for multi head attention, e.g. d_model=64, head_dim=16(then only the shape of Q,K,V changes)
        super().__init__()

        self.query = nn.Linear(in_features=d_model,
                               out_features=head_dim)
        self.key = nn.Linear(in_features=d_model,
                             out_features=head_dim)
        self.value = nn.Linear(in_features=d_model,
                               out_features=head_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores/(K.size(-1)**(0.5))

        mask = torch.tril(torch.ones(x.size(1), x.size(1), device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        output = attention_weights @ V

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        head_dim = d_model // num_heads

        self.heads = nn.ModuleList([SelfAttention(d_model, head_dim) for _ in range(num_heads)])
        self.output_projection = nn.Linear(in_features = d_model,
                                           out_features= d_model)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)

        output = self.output_projection(combined)

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(in_features=d_model, out_features=4*d_model),
                                     nn.GELU(),
                                     nn.Linear(in_features=4*d_model, out_features=d_model))

    def forward(self, x):
        return self.network(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)

        self.ffn = FeedForward(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))

        return x

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])

    def forward(self, x):
        for block in self.layers:
            x = block(x)

        return x

class LanguageModelHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()

        self.norm = LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.norm(x)
        return self.output(x)


class GPTModel(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, num_heads, num_layers):
        super().__init__()

        self.embeddings = InputEmbedding(vocab_size, block_size, d_model)
        self.transformer = Transformer(d_model, num_heads, num_layers)
        self.head = LanguageModelHead(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.embeddings(token_ids)
        x = self.transformer(x)
        x = self.head(x)

        return x