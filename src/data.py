from src.tokenizer import tokenize, build_vocab, encode, create_sequences
import torch
from torch.utils.data import DataLoader
from src.dataset import TextDataset
from src.config import BLOCK_SIZE, BATCH_SIZE

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

def create_dataloader(file_path, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE):
    inputs, targets, vocab = prepare_data(file_path)
    input_chunks, target_chunks = create_chunks(inputs, targets, block_size)
    
    dataset = TextDataset(input_chunks, target_chunks)

    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        drop_last=True)

    return loader, vocab

def create_train_val_dataloaders(file_path, block_size=BLOCK_SIZE, batch_size=BATCH_SIZE, val_ratio=0.2):
    tokens = load_tokens(file_path)

    split_index = int((1-val_ratio)*len(tokens))

    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]

    vocab = build_vocab(train_tokens)

    train_ids = encode(train_tokens, vocab)
    val_ids = encode(val_tokens, vocab)

    train_inputs, train_targets = create_sequences(train_ids)
    val_inputs, val_targets = create_sequences(val_ids)

    train_inputs = torch.tensor(train_inputs, dtype=torch.long)
    train_targets = torch.tensor(train_targets, dtype=torch.long)
    val_inputs = torch.tensor(val_inputs, dtype=torch.long)
    val_targets = torch.tensor(val_targets, dtype=torch.long)

    train_input_chunks, train_target_chunks = create_chunks(train_inputs, train_targets, block_size)
    val_input_chunks, val_target_chunks = create_chunks(val_inputs, val_targets, block_size)

    train_dataset = TextDataset(train_input_chunks, train_target_chunks)
    val_dataset = TextDataset(val_input_chunks, val_target_chunks)

    train_loader = DataLoader(train_dataset,
                              batch_size = batch_size, 
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size,
                            shuffle=False, 
                            drop_last=False)

    return train_loader, val_loader, vocab