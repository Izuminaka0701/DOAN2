import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Chuẩn bị dữ liệu MNIST & CIFAR-10
transform_mnist = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform_mnist)
test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform_mnist)
train_cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
test_cifar = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)

train_loader_mnist = DataLoader(train_mnist, batch_size=64, shuffle=True)
test_loader_mnist = DataLoader(test_mnist, batch_size=1, shuffle=False)
train_loader_cifar = DataLoader(train_cifar, batch_size=64, shuffle=True)
test_loader_cifar = DataLoader(test_cifar, batch_size=1, shuffle=False)

# Mô hình CNN cho MNIST & CIFAR-10
class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Hàm huấn luyện chung
def train_model(model, train_loader, num_epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_history = []
    plt.ion()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        plt.clf()
        plt.plot(loss_history, label='Loss')
        plt.legend()
        plt.pause(0.1)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    plt.ioff()
    plt.show()

# Huấn luyện MNIST & CIFAR-10
print("Training on MNIST")
mnist_model = CNN(input_channels=1)
train_model(mnist_model, train_loader_mnist)

print("Training on CIFAR-10")
cifar_model = CNN(input_channels=3)
train_model(cifar_model, train_loader_cifar)

# # Mô hình LSTM cho xử lý ngôn ngữ tự nhiên (NLP)
# class LSTMClassifier(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
#         super(LSTMClassifier, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, output_dim)
    
#     def forward(self, x):
#         x = self.embedding(x)
#         x, _ = self.lstm(x)
#         x = self.fc(x[:, -1, :])
#         return x

# Ví dụ: Huấn luyện LSTM trên dữ liệu văn bản (bỏ qua phần load dữ liệu)
# vocab_size, embedding_dim, hidden_dim, output_dim = 5000, 128, 256, 2
# lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
# train_model(lstm_model, train_loader_text)
