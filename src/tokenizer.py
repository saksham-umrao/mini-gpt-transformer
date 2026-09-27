import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    
    return text.split()

def build_vocab(tokens):
    vocab = {"<UNK>": 0}

    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab

def encode(tokens, vocab):
    unk_id = vocab["<UNK>"]

    return [vocab.get(token, unk_id) for token in tokens]

def create_sequences(token_ids):
    inputs = token_ids[:-1]
    targets =  token_ids[1:]

    return inputs, targets