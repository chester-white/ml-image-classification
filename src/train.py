import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model import Model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 4
batch_size = 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5000 == 0:
            print(loss.item())

with torch.no_grad():
    correct = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / len(test_dataset)
    print(f'Accuracy: {accuracy} %')
