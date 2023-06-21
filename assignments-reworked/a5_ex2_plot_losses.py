import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

def plot_losses(train_losses: list, eval_losses: list):
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, eval_losses, label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error Loss')
    plt.legend()
    plt.show()

def training_loop(network: nn.Module,
                  train_data: torch.utils.data.Dataset,
                  eval_data: torch.utils.data.Dataset,
                  num_epochs: int,
                  show_progress: bool = False) -> tuple[list, list]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device '{device}'.")

    optimizer = optim.Adam(network.parameters(), lr=0.001) # best results with Adam
    #optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=32, shuffle=False)
    network.to(device)

    train_losses = []
    eval_losses = []

    # preparing for early stopping
    best_eval_loss = float('inf')
    consecutive_no_improvement = 0

    for epoch in range(num_epochs):
        network.train()
        epoch_train_loss = 0.0
        num_train_batches = len(train_loader)

        if show_progress:
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training")

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # forward pass
            outputs = network(inputs).squeeze() # otherwise warning AND poor performance (dimensions must match)
            loss = nn.MSELoss()(outputs, targets)

            # backward pass/optimization
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        # average training loss for the epoch
        epoch_train_loss /= num_train_batches
        train_losses.append(epoch_train_loss)

        network.eval()
        epoch_eval_loss = 0.0
        num_eval_batches = len(eval_loader)

        if show_progress:
            eval_loader = tqdm(eval_loader, desc=f"Epoch {epoch + 1} Evaluation")

        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # forward pass
                outputs = network(inputs).squeeze() # otherwise warning AND poor performance (dimensions must match)
                loss = nn.MSELoss()(outputs, targets)

                epoch_eval_loss += loss.item()

        # average evaluation loss for the epoch
        epoch_eval_loss /= num_eval_batches
        eval_losses.append(epoch_eval_loss)

        # now the stopping check
        if epoch_eval_loss < best_eval_loss:
            best_eval_loss = epoch_eval_loss
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        if consecutive_no_improvement >= 3:
            break

    return train_losses, eval_losses

if __name__ == "__main__":
    from a4_ex1 import SimpleNetwork
    from dataset import get_dataset

    torch.random.manual_seed(0)
    train_data, eval_data = get_dataset()
    network = SimpleNetwork(32, 128, 1)
    train_losses, eval_losses = training_loop(network, train_data, eval_data, num_epochs=100)
    #for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
    #   print(f"Epoch: {epoch} --- Train loss: {tl:7.2f} --- Eval loss: {el:7.2f}")
    plot_losses(train_losses, eval_losses)
