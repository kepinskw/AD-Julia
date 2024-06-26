import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import tracemalloc
import time


batch_size = 100
learning_rate = 15e-3
epochs = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  
])

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(14 * 14, 64, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 4, 196)  
        h0 = torch.zeros(1, x.size(0), 64)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model = RNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def loss_and_accuracy(loader):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            y_hat = model(x)
            loss = criterion(y_hat, y)
            _, predicted = torch.max(y_hat, 1)
            accuracy = (predicted == y).float().mean().item() * 100
    return loss.item(), accuracy


train_log = []
main_time = time.time()
tracemalloc.start()  
for epoch in range(epochs):
    start_time = time.time()
    
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    
    train_loss, train_acc = loss_and_accuracy(train_loader)
    test_loss, test_acc = loss_and_accuracy(test_loader)
    end_time = time.time()
    elapsed_time_ep = end_time - start_time
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print(f"Epoch time: {elapsed_time_ep} seconds")
    train_log.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'test_loss': test_loss,
        'test_acc': test_acc
    })

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current/2**30}GiB; Peak was {peak/2**30}GiB")
tracemalloc.stop()  # Stop tracing memory allocations

main_endtime = time.time()
elapsed_time = main_endtime - main_time
print(f"Total time taken: {elapsed_time} seconds")
