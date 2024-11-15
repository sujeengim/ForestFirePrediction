import torch
import os
from tqdm import tqdm
import time
import wandb
from config import device

class Trainer:
    """Trainer class that takes care of training and validation passes."""

    def __init__(
        self,
        model,
        optimizer,
        lr,
        epochs=10,
        precision="fp32",
        device=device,
        use_wandb=True,
        use_ipex=False,
    ):
        self.use_ipex = use_ipex
        self.use_wandb = use_wandb
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.epochs = epochs
        self.lr = lr
        self.precision = precision
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", verbose=True
        )

    def forward_pass(self, inputs, labels):
        """Perform forward pass of models with `inputs`,
        calculate loss and accuracy and return it.
        """
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        preds = outputs.argmax(dim=1, keepdim=True)
        correct = preds.eq(labels.view_as(preds)).sum().item()
        total = labels.numel()
        return loss, correct, total, preds, labels  # preds, labels 추가

    # *************************** Exercise 2 ***************************************
    def _to_ipex(self, dtype=torch.float32):
        """convert model memory format to channels_last to IPEX format."""
        self.model.train()
        self.model = self.model.to(memory_format=torch.channels_last)
        self.model, self.optimizer = ipex.optimize(
            self.model, optimizer=self.optimizer, dtype=torch.float32
        )
    # ******************************************************************************

    def train_one_epoch(self, train_dataloader):
        """Training loop for one epoch, return epoch loss, accuracy, predictions, and labels."""
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(train_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # Forward pass (precision option remains the same)
            if self.precision == "bf16":
                with getattr(torch, f"{self.device.type}.amp.autocast")():
                    loss, correct, batch_size, preds, labels = self.forward_pass(inputs, labels)
            else:
                loss, correct, batch_size, preds, labels = self.forward_pass(inputs, labels)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_size

            # Flatten predictions to ensure correct format (if it's probabilistic output, take max or threshold)
            preds = torch.argmax(preds, dim=1) if preds.ndim > 1 else preds  # Use argmax for multi-class or keep if binary
            all_preds.append(preds)
            all_labels.append(labels)

            acc = total_correct / total_samples
            if self.use_wandb:
                wandb.log(
                    {
                        "Training Loss": total_loss / len(train_dataloader),
                        "Training Acc": acc,
                    }
                )

        return total_loss / len(train_dataloader), acc, torch.cat(all_preds), torch.cat(all_labels)

    @torch.no_grad()  # Ensure this decorator is at the same indentation level as the function definition
    def validate_one_epoch(self, valid_dataloader):
        """Validation loop for one epoch, return epoch loss, accuracy, predictions, and labels."""
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(valid_dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            loss, correct, batch_size, preds, labels = self.forward_pass(inputs, labels)

            # Update metrics
            total_loss += loss.item()
            total_correct += correct
            total_samples += batch_size

            # Flatten predictions for ROC curve compatibility
            preds = torch.argmax(preds, dim=1) if preds.ndim > 1 else preds  # Use argmax if multi-class, else keep
            all_preds.append(preds)
            all_labels.append(labels)

            acc = total_correct / total_samples
            if self.use_wandb:
                wandb.log(
                    {
                        "Validation Loss": total_loss / len(valid_dataloader),
                        "Validation Acc": acc,
                    }
                )

        self.scheduler.step(total_loss / len(valid_dataloader))
        return total_loss / len(valid_dataloader), acc, torch.cat(all_preds), torch.cat(all_labels)


    def fine_tune(self, train_dataloader, valid_dataloader):
        if self.use_ipex:
            self._to_ipex()
        if self.use_wandb:
            import os
            print(os.environ["WANDB_DIR"])
            wandb.init(project="FireFinder", name="FireFinder", dir="./wandb_logs")
        for epoch in range(self.epochs):
            t_epoch_start = time.time()
            t_epoch_loss, t_epoch_acc, train_preds, train_labels = self.train_one_epoch(train_dataloader)
            v_epoch_loss, v_epoch_acc, valid_preds, valid_labels = self.validate_one_epoch(valid_dataloader)
            t_epoch_end = time.time()
            print(
                f"\n📅 Epoch {epoch+1}/{self.epochs}:\n"
                f"\t🏋️‍♂️ Training step:\n"
                f"\t - 🎯 Loss: {t_epoch_loss:.4f}"
                f", 📈 Accuracy: {t_epoch_acc:.4f}\n"
                f"\t🧪 Validation step:\n"
                f"\t - 🎯 Loss: {v_epoch_loss:.4f}"
                f", 📈 Accuracy: {v_epoch_acc:.4f}\n"
                f"⏱️ Time: {t_epoch_end - t_epoch_start:.4f} sec\n"
            )
            df = pd.DataFrame({
               'epoch': EPOCHS,
               'Train Loss': t_epoch_loss,
               'Validation Loss': v_epoch_loss,
               'Train Accuracy': t_epoch_acc,
               'Validation Accuracy': v_epoch_acc
            })
            
            if self.use_wandb:
                wandb.log(
                    {
                        "Train Loss": t_epoch_loss,
                        "Train Acc": t_epoch_acc,
                        "Valid Loss": v_epoch_loss,
                        "Valid Acc": v_epoch_acc,
                        "Time": t_epoch_end - t_epoch_start,
                    }
                )

        if self.use_wandb:
            wandb.finish()
        return int(v_epoch_acc * 100)
