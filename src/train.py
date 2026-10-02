from src.utils import set_seed
from src.data import create_train_val_dataloaders
from src.model import GPTModel
from src.training import train_model
from src.config import BLOCK_SIZE, BATCH_SIZE


set_seed(27)

train_loader, val_loader, vocab = create_train_val_dataloaders("data/input.txt", 
                                                               block_size=BLOCK_SIZE,
                                                               batch_size=BATCH_SIZE)

model = GPTModel(vocab_size=len(vocab),
                 block_size=BLOCK_SIZE,
                 d_model=64,
                 num_heads=4,
                 num_layers=4)

train_model(model,
            train_loader,
            val_loader,
            epochs=100,
            learning_rate=3e-4)