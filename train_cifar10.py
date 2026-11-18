
# ============================================================
# 1. IMPORTS
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Select hardware: Apple GPU (MPS), NVIDIA GPU (CUDA), or CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Hyperparameters: values selected before training.
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

print("Device:", device)


# ============================================================
# 3. DATASET AND DATALOADERS
# ============================================================

# Convert images to float32 tensors with shape (C, H, W).
# Pixel values are scaled from [0, 255] to [0, 1].
transform = transforms.ToTensor()

# Download CIFAR-10.
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Split the original 50,000 training images:
# 49,000 for training; 1,000 for validation.
train_set = Subset(train_dataset, range(0, 49000))
val_set = Subset(train_dataset, range(49000, 50000))

# DataLoader supplies (images, labels) in minibatches.
# shuffle=True changes training sample order each epoch.
loader_train = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)

loader_val = DataLoader(
    val_set,
    batch_size=batch_size,
    shuffle=False
)

loader_test = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 4. DEFINE MODEL ARCHITECTURE
# ============================================================

# nn.Sequential constructs a model from an ordered list of layers.
# It does NOT execute the forward pass until model(x) is called.
#
# Convolution layers learn spatial features.
# BatchNorm normalizes feature activations.
# ReLU introduces nonlinearity.
# MaxPool reduces spatial resolution by a factor of 2.

model = nn.Sequential(

    # BLOCK 1
    # Input: (N, 3, 32, 32)

    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    # (N, 3, 32, 32) -> (N, 32, 32, 32)
    # Learnable weights: (32, 3, 3, 3)

    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),

    nn.MaxPool2d(kernel_size=2),
    # (N, 32, 32, 32) -> (N, 32, 16, 16)


    # BLOCK 2
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),

    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),

    nn.MaxPool2d(kernel_size=2),
    # (N, 64, 16, 16) -> (N, 64, 8, 8)


    # BLOCK 3
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),

    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),

    nn.MaxPool2d(kernel_size=2),
    # (N, 128, 8, 8) -> (N, 128, 4, 4)


    # CLASSIFIER
    nn.Flatten(),
    # (N, 128, 4, 4) -> (N, 2048)

    nn.Linear(128 * 4 * 4, 128),
    # (N, 2048) -> (N, 128)

    nn.ReLU(),

    nn.Linear(128, 10)
    # (N, 128) -> (N, 10): 10 class scores
)

# Move model parameters and buffers to the selected device.
model = model.to(device)


# ============================================================
# 5. DEFINE LOSS AND OPTIMIZER
# ============================================================

# Cross-entropy compares class scores with correct labels.
# Input: scores (N, 10), labels (N,)
# Raw logits are expected; do not add Softmax to the model.
criterion = nn.CrossEntropyLoss()

# Adam updates all trainable parameters registered in the model.
# model.parameters() supplies references to those parameters.
# lr=1e-3 is the optimizer's learning-rate hyperparameter.
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 6. DEFINE EVALUATION FUNCTION
# ============================================================

def check_accuracy(loader, model):
    # Switch BatchNorm to inference behavior.
    model.eval()

    num_correct = 0
    num_samples = 0

    # Evaluation does not need a computation graph.
    with torch.no_grad():
        for x, y in loader:

            # Move minibatch to the same device as the model.
            x = x.to(device)
            y = y.to(device)

            # Forward pass: (N, 3, 32, 32) -> (N, 10)
            scores = model(x)

            # Select the class with the highest score.
            preds = scores.argmax(dim=1)

            # Accumulate correct predictions and sample count.
            num_correct += (preds == y).sum().item()
            num_samples += y.size(0)

    return 100 * num_correct / num_samples


# ============================================================
# 7. TRAINING LOOP
# ============================================================

train_losses = []
val_accuracies = []

for epoch in range(num_epochs):

    # Enable training behavior for BatchNorm.
    model.train()

    running_loss = 0.0

    # One epoch = one full pass through loader_train.
    for x, y in loader_train:

        # A minibatch of input images and their class labels.
        # x: (N, 3, 32, 32)
        # y: (N,)
        x = x.to(device)
        y = y.to(device)

        # STEP 1: Clear gradients from the previous iteration.
        optimizer.zero_grad()

        # STEP 2: Forward pass.
        # Sequential runs each layer in order.
        scores = model(x)

        # STEP 3: Calculate classification loss.
        loss = criterion(scores, y)

        # STEP 4: Backward pass.
        # Autograd computes gradients for trainable parameters.
        loss.backward()

        # STEP 5: Update parameters using Adam.
        optimizer.step()

        # Accumulate minibatch loss, weighted by batch size.
        running_loss += loss.item() * x.size(0)

    # Average training loss across all training images.
    epoch_loss = running_loss / len(loader_train.dataset)
    train_losses.append(epoch_loss)

    # Evaluate validation accuracy after this epoch.
    val_accuracy = check_accuracy(loader_val, model)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{num_epochs} | "
        f"Train loss: {epoch_loss:.4f} | "
        f"Validation accuracy: {val_accuracy:.2f}%"
    )


# ============================================================
# 8. VISUALIZE TRAINING RESULTS
# ============================================================

epochs = range(1, num_epochs + 1)

# Average training loss by epoch
plt.figure(figsize=(8, 4))
plt.plot(epochs, train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("CIFAR-10 Training Loss")
plt.grid(True, alpha=0.3)
plt.show()

# Validation accuracy by epoch
plt.figure(figsize=(8, 4))
plt.plot(epochs, val_accuracies, marker="o")
plt.axhline(70, linestyle="--", label="70% target")
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy (%)")
plt.title("CIFAR-10 Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================

# Use the test set only after architecture and tuning are done.
# This evaluates the final weights after epoch 10.
best_model = model

test_accuracy = check_accuracy(loader_test, best_model)

print(f"Final test accuracy: {test_accuracy:.2f}%")


# ============================================================
# 10. SAVE MODEL WEIGHTS
# ============================================================

# Save model parameters for later reuse without retraining.
torch.save(
    best_model.state_dict(),
    "cifar10_convnet.pth"
)

print("Saved model weights to cifar10_convnet.pth")
