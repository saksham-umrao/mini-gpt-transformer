import torch


def save_checkpoint(model, 
                    optimizer, 
                    scheduler,epoch, 
                    best_val_loss, 
                    file_path):
    
    checkpoint = {"epoch": epoch,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": scheduler.state_dict(),
                  "best_val_loss": best_val_loss}

    torch.save(checkpoint, file_path)

def load_checkpoint(model,
                    optimizer,
                    scheduler,
                    file_path):
    """used when we want to resume training of the model at some point of time."""
    checkpoint = torch.load(file_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]

    return model, optimizer, scheduler, start_epoch, best_val_loss

def load_model_for_inference(model, file_path):
    """used when we want to just load the model for inference, that's why only returning the model and not its whole state of training."""
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model