import torch
from torch import nn, optim

from utils import save_experiment, save_checkpoint
from data import prepare_data
from vit import ViTForClassfication
import yaml
from tqdm import tqdm
import argparse
from transformers import get_linear_schedule_with_warmup


parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", type=str, 
                    default='configs/test1.yaml', required=True)
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
# These are not hard constraints, but are used to prevent misconfigurations
assert config["hidden_size"] % config["num_attention_heads"] == 0
# assert config['intermediate_size'] == 4 * config['hidden_size']
assert config['image_size'] % config['patch_size'] == 0

class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_fn, exp_name, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.exp_name = exp_name
        self.device = device

    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):
            self.iepoch = i
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            current_lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, "
                  f"Test loss: {test_loss:.4f},"
                  f"Accuracy: {accuracy:.4f},"
                  f"Lr: {current_lr}")
            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:
                print('\tSave checkpoint at epoch', i+1)
                save_checkpoint(self.exp_name, self.model, i+1)
        # Save the experiment
        save_experiment(self.exp_name, config, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in tqdm(trainloader):
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch
                
                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def main():
    # Training parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = config['exp_name']
    epochs = config['epochs']
    batch_size = config['batch_size']
    lr = config['init_lr']
    save_model_every_n_epochs = config['save_model_every']
    init_ckpt = config['init_ckpt']

    # Load the CIFAR10 dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer
    model = ViTForClassfication(
        hidden_size=config['hidden_size'],
        num_layers=config['num_hidden_layers'],
        num_heads=config['num_attention_heads'],
        mlp_ratio=config['mlp_ratio'],
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        dropout_prob=config['hidden_dropout_prob'],
        bias=config['qkv_bias'],
        num_classes=config['num_classes'],
        initializer_range=config['initializer_range'],
    )
    if init_ckpt is not None:
        ckpt = torch.load(init_ckpt, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {init_ckpt}")
    total_steps = len(trainloader) * epochs
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1*total_steps, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, lr_scheduler, loss_fn, exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, save_model_every_n_epochs=save_model_every_n_epochs)


if __name__ == "__main__":
    # pass
    main()