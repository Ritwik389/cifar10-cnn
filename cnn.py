import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import DeepNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Training on: {device}")


stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(*stats),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

net = DeepNet().to(device)


print(f"Total Params: {sum(p.numel() for p in net.parameters()):,}")


EPOCHS = 100
MAX_LR = 0.05 
weight_decay = 1e-4


criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.SGD(net.parameters(), lr=MAX_LR, momentum=0.9, weight_decay=weight_decay)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,
    epochs=EPOCHS,
    steps_per_epoch=len(trainloader)
)

print(f"Starting Training for {EPOCHS} epochs...")


for epoch in range(EPOCHS):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()


    if (epoch + 1) % 5 == 0 or epoch == 0:
        train_acc = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {running_loss/len(trainloader):.4f} | Acc: {train_acc:.2f}% | LR: {current_lr:.6f}")



print("Running TTA Evaluation...")
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)


        outputs1 = net(inputs)

        inputs_flipped = torch.flip(inputs, [3])
        outputs2 = net(inputs_flipped)


        outputs_avg = (outputs1 + outputs2) / 2.0

        _, predicted = torch.max(outputs_avg.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"FINAL TEST ACCURACY (w/ TTA): {100 * correct / total:.2f}%")