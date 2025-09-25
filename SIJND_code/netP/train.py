import SI_dataset
import c_patches
from fr_model import FRnet
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import torchvision.transforms as transforms
from tqdm import tqdm  # For progress bars
import copy
import matplotlib.pyplot as plt  # For plotting
import pandas as pd  # For handling Excel files


def cls(y_pred):
    """Convert model outputs to binary class predictions (0/1) using sigmoid threshold"""
    m = nn.Sigmoid()
    lei = m(y_pred)  # Apply sigmoid activation
    Y = torch.zeros(len(lei))  # Initialize output tensor
    for i in range(len(lei)):
        if lei[i] > 0.5:  # Threshold at 0.5 for binary classification
            Y[i] = 1
        else:
            Y[i] = 0
    return Y


def train_one_epoch(epoch, epochs, model, train_loader, vail_loader, device, optimizer, criterion):
    """Train model for one epoch and validate on test set"""
    correct = 0
    total = 0
    sum_loss = 0
    model.train()  # Set model to training mode

    # Training loop with progress bar
    loop = tqdm(train_loader, desc='Train')
    for x, y in loop:
        x, y = x.to(device), y.to(device)  # Move data to device (GPU/CPU)
        y_pred = model(x)  # Forward pass
        y_pred = y_pred.squeeze(1)  # Remove extra dimension
        loss = criterion(y_pred, y.float())  # Calculate loss
        y_pred_class = cls(y_pred)  # Get class predictions

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            y_pred_class = y_pred_class.to(device)
            correct += (y_pred_class == y).sum().item()
            total += y.size(0)
            sum_loss += loss.item()
            running_loss = sum_loss / total
            running_acc = correct / total

        # Update progress bar
        loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
        loop.set_postfix(loss=running_loss, acc=running_acc)

    epoch_loss = sum_loss / total
    epoch_acc = correct / total

    # Validation phase
    test_correct = 0
    test_total = 0
    test_sum_loss = 0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        loop2 = tqdm(vail_loader, desc='Test')
        for x, y in loop2:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_pred = y_pred.squeeze(1)
            loss = criterion(y_pred, y.float())
            y_pred_class = cls(y_pred)
            y_pred_class = y_pred_class.to(device)
            test_correct += (y_pred_class == y).sum().item()
            test_total += y.size(0)
            test_sum_loss += loss.item()
            test_running_loss = test_sum_loss / test_total
            test_running_acc = test_correct / test_total

            loop2.set_postfix(loss=test_running_loss, acc=test_running_acc)

    test_epoch_loss = test_sum_loss / test_total
    test_epoch_acc = test_correct / test_total

    # Save model checkpoint
    torch.save(model, model_path + 'model_' + str(epoch) + '.pth')

    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc


def train_fuc(n, train_path, vail_path, model_path):
    """Main training function for k-fold cross-validation"""
    # Load datasets
    train_dataset = SI_dataset.Mydataset(excel_file=train_path, mode='train')
    vail_dataset = SI_dataset.Mydataset(excel_file=vail_path, mode='vail')
    batch_size = 8

    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    vail_loader = DataLoader(
        dataset=vail_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # Set device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FRnet()  # Initialize model
    model.to(device=device)  # Move model to device

    # Set optimizer and loss function
    lr = 0.00001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss

    epochs = 100

    # Initialize tracking variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc = train_one_epoch(
            epoch, epochs, model, train_loader, vail_loader, device, optimizer, criterion
        )

        # Store metrics
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)

        # Save best model based on validation performance
        if len(test_loss) > 1:
            if test_loss[-1] < test_loss[-2] and test_acc[-1] > best_acc:
                best_acc = test_acc[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_path + 'best_model.pth')

    # Save metrics to Excel
    dfData = {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc
    }
    df = pd.DataFrame(dfData)
    df.to_excel('Kfold/acc_loss/' + str(n) + '.xlsx', index=False)

    # Plot accuracy and loss curves
    x1 = range(1, epochs + 1)
    plt.ion()  # Interactive mode
    plt.subplot(2, 1, 1)
    plt.plot(x1, train_acc, color='g', label='train_acc')
    plt.plot(x1, test_acc, color='b', label='test_acc')
    plt.legend()
    plt.title('Accuracy vs. epoches')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(x1, train_loss, color='g', label='train_loss')
    plt.plot(x1, test_loss, color='b', label='test_loss')
    plt.legend()
    plt.xlabel('Loss vs. epoches')
    plt.ylabel('Loss')

    plt.savefig('Kfold/figs/accuracy_loss' + str(n) + '.jpg')
    plt.pause(10)  # Display plot briefly
    plt.close()


# Main execution
PATH = 'Kfold/'
train_path = 'Dataset/data3/JND_train3.xlsx'
vail_path = 'Dataset/data3/JND_vail3.xlsx'
model_path = 'Kfold/model_3/'
train_fuc(3, train_path, vail_path, model_path)  # Run training for fold 3