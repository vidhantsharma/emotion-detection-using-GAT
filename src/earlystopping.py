import torch

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, save_path="best_model.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.save_path = save_path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)  # Save initial model
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)  # Update best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered")
                return True
        return False

    def save_checkpoint(self, model):
        """Saves the model when validation loss decreases."""
        torch.save(model.state_dict(), self.save_path)
        if self.verbose:
            print(f"Validation loss decreased, saving model to {self.save_path}")
