import os
import pandas as pd
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import glob
import torch.nn as nn
from torch.optim import Adam
import pathlib
import torch.nn.functional as F
import sklearn
from sklearn.metrics import f1_score

#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.cuda.is_available())
print(torch.version.cuda)
#transform
train_transformer = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5]),
    transforms.RandomErasing(p=0.1)
])

test_transformer = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                         [0.5,0.5,0.5])
])
#try normalization with : [0.485, 0.456, 0.406],
#                         [0.229, 0.224, 0.225]
#data folder paths
train_path='/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge/train_images'
test_path='/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge/test_images'
csv_path = '/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge/train_labels.csv'
classes_path = '/kaggle/input/competitions/shift-guard-10-robust-image-classification-challenge/classes.txt'
#custom train dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, classes_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load CSV
        self.data = pd.read_csv(csv_file, dtype={'id': str})
        
        # Load class names
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx]['id'])
        label_name = self.data.iloc[idx]['label']
        
        img_name = img_id + ".png"
        label = self.class_to_idx[label_name]
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
#custom test dataset
class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, img_name

full_dataset = CustomImageDataset(
    img_dir=train_path,
    csv_file=csv_path,
    classes_file=classes_path,
    transform=train_transformer
)
print(len(full_dataset));
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
indices = torch.randperm(len(full_dataset)).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = torch.utils.data.Subset(
    CustomImageDataset(train_path, csv_path, classes_path, transform=train_transformer),
    train_indices
)

val_dataset = torch.utils.data.Subset(
    CustomImageDataset(train_path, csv_path, classes_path, transform=test_transformer),
    val_indices
)

test_dataset = TestDataset(
    img_dir=test_path,
    transform=test_transformer
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)
images, labels = next(iter(train_loader))
print(images.shape)
print(labels[:10])
print(images.min(), images.max())
print(labels.min(), labels.max())


#defining convNET
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = F.relu(out, inplace=True)

        return out


class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layer1 = nn.Sequential(
            ResidualBlock(3, 32,stride=1),
            ResidualBlock(32, 32),
            nn.MaxPool2d(2)
        )
        #output size :(32, 16, 16)

        self.layer2 = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 64),
            nn.MaxPool2d(2)
        )
        #output size :(64, 8, 8)
        self.layer3 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2)
            
        )
        #output size :(128, 4, 4)
        self.layer4 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256)
        )
        #output size :(256, 4, 4)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        #output size :(256, 1, 1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)   # 32 channels
        x = self.layer2(x)   # 64 channels
        x = self.layer3(x)   # 128 channels
        x = self.layer4(x)   # 256 channels

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        x = self.fc(x)

        return x

model = CustomResNet().to(device)
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

#Optmizer, scheduler and loss function
#Optmizer, scheduler and loss function
optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=3,
)
loss_function=nn.CrossEntropyLoss(label_smoothing=0.1)
num_epochs=100
#calculating the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.png'))
test_count=len(glob.glob(test_path+'/**/*.png'))
print(train_count,test_count)
#model training
best_accuracy = 0.0
best_f1=0.0
for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        train_correct += torch.sum(preds == labels)

    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct.double() / len(train_loader.dataset)
    
     # ================= VALIDATION =================
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average='macro')

    # ================= LOG =================
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {train_loss:.4f}, "
          f"Train Acc: {train_accuracy:.4f}, "
          f"Val F1: {val_f1:.4f}")

    # ================= SCHEDULER =================
    scheduler.step(val_f1)

    # ================= SAVE BEST =================
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved")

model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
model.eval()

preds = []

# ✅ FIX 1: get mapping properly
idx_to_class = full_dataset.idx_to_class   # correct mapping

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)   # ✅ FIX 2: device (not DEVICE)

        outputs = model(images)
        pred = outputs.argmax(dim=1).cpu().numpy()
        preds.extend(pred)

# Convert predictions to labels
labels = [idx_to_class[p] for p in preds]

# ✅ FIX 3: KEEP ORIGINAL IDs (NO stripping zeros)
test_ids = [img.replace(".png", "").replace(".jpg", "") 
            for img in test_dataset.images]

# Create submission
submission = pd.DataFrame({
    "id": test_ids,
    "label": labels
})

# ✅ IMPORTANT: sort lexicographically (as strings)
submission = submission.sort_values("id")

# Save
submission.to_csv("/kaggle/working/submission.csv", index=False)

print("✅ submission.csv created successfully!")
