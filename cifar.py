import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def main():
    # torch config
    torch.manual_seed(42)
    torch.multiprocessing.freeze_support()

    # Import CIFAR-10 dataset
    print('Importing data')
    dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
    test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

    # Prepare for training
    print('Preparing data')
    validation_size = 5000
    train_size = len(dataset) - validation_size
    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size])

    batch_size = 128

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size*2, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size * 2, num_workers=2, pin_memory=True
    )

    # Generate Model
    print('Generating model')
    device = get_default_device()

    input_size = 3*32*32
    output_size = 10
    model = to_device(CIFAR10Model(input_size=input_size, hidden_size=1028,
                                   output_size=output_size), device)

    # Train Model
    epochs = 15
    print(f'Training model for {epochs} epoch(s)')
    history = [evaluate(model, validation_loader)]
    history += fit(epochs, 0.1, model, train_loader, validation_loader)
    plot_accuracies(history)
    plot_losses(history)

    # Evaluate model
    print('Evaluating Model')
    results = evaluate(model, test_loader)
    print(results)


class ImageClassifierBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = functional.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = functional.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'validation_loss': loss.detach(), 'validation_accuracy': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['validation_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['validation_accuracy'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'validation_loss': epoch_loss.item(), 'validation_accuracy': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], validation_loss: { result['validation_loss']:.4f}, validation_accuracy: {result['validation_accuracy']:.4f}, time_taken: {result['time_taken']:.4f}s")


class CIFAR10Model(ImageClassifierBase):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, output_size)

    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.linear1(out)
        out = functional.relu(out)
        out = self.linear2(out)
        out = functional.relu(out)
        out = self.linear3(out)
        out = functional.relu(out)
        out = self.linear4(out)
        out = functional.relu(out)
        out = self.linear5(out)
        out = functional.relu(out)
        out = self.linear6(out)
        out = functional.relu(out)
        return out

 # === Helpers ====


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def evaluate(model, validation_loader):
    outputs = [model.validation_step(batch) for batch in validation_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, validation_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        start = time.time()
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, validation_loader)
        end = time.time()
        result['time_taken'] = end - start
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def plot_losses(history):
    losses = [x['validation_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()


def plot_accuracies(history):
    accuracies = [x['validation_accuracy'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()


def get_default_device():
    if torch.cuda.is_available():
        print("Running on GPU")
        return torch.device('cuda')
    else:
        print("Running on CPU")
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    # === Entrypoint ===
if __name__ == '__main__':
    main()
