from src.tokenizer import tokenize, build_vocab, encode, create_sequences
import torch

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text

def load_tokens(file_path):
    text = load_text(file_path)
    tokens = tokenize(text)

    return tokens

def build_corpus_vocab(file_path):
    tokens = load_tokens(file_path)
    vocab = build_vocab(tokens)

    return vocab

def encode_corpus(file_path):
    tokens = load_tokens(file_path)
    vocab = build_vocab(tokens)
    token_ids = encode(tokens, vocab)

    return token_ids, vocab


def prepare_data(file_path):
    token_ids, vocab = encode_corpus(file_path)
    inputs, targets = create_sequences(token_ids)

    inputs = torch.tensor(inputs, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return inputs, targets, vocab

def create_chunks(inputs, targets, block_size):
    """creates overlapping/sliding chunks of the encoded inputs and targets."""
    input_chunks = []
    target_chunks = []

    for i in range(0, len(inputs)-block_size+1):
        input_chunks.append(inputs[i:i+block_size])
        target_chunks.append(targets[i:i+block_size])

    return input_chunks, target_chunks