import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model().to(device)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epochs = 70
losses = []
losses_table = PrettyTable()
losses_table.field_names = ['Epoch', 'Loss']

for i in range(epochs):
    losses_sum = 0
    for j, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        loss = criterion(predictions, labels)
        losses_sum += loss.detach().numpy()
        if j == len(train_loader) - 1:
            losses.append(losses_sum / len(train_loader))
            if i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or (i + 1) % 5 == 0:
                losses_table.add_row([i + 1, losses_sum / len(train_loader)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(losses_table)

correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {(correct / len(test_data) * 100):.2f}% ({correct}/{len(test_data)})')

plt.get_current_fig_manager().set_window_title('Training')
plt.plot(range(epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
