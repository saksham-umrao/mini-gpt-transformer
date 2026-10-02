import torch
import torch.nn.functional as F
import copy
from src.checkpoint import save_checkpoint
import math

def calculate_loss(logits, targets):
    batch_size, sequence_length, vocab_size = logits.shape

    logits = logits.view(batch_size * sequence_length, vocab_size)
    targets = targets.view(batch_size * sequence_length)

    loss = F.cross_entropy(logits, targets)

    return loss

def create_optimizer(model, learning_rate=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return optimizer

def evaluate_model(model, val_loader):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            logits = model(inputs)

            loss = calculate_loss(logits, targets)
            total_loss += loss.item()

    average_loss = total_loss/len(val_loader)

    return average_loss



def train_model(model, train_loader, val_loader, epochs=100, learning_rate=3e-4, early_stopping_patience=10):
    optimizer = create_optimizer(model, learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           factor=0.5,
                                                           patience=5)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for inputs, targets in train_loader:
            logits = model(inputs)
            loss = calculate_loss(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        val_loss = evaluate_model(model, val_loader)

        val_perplexity = calculate_perplexity(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, "best_model.pt")
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch + 1}/{epochs}, Training_Loss: {train_loss:.4f}, Val_Loss: {val_loss:.4f}, Val_PPL: {val_perplexity:.4f}, LR: {current_lr:.6f}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

def calculate_perplexity(loss):
    return math.exp(loss)

