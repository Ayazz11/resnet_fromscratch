import os
import pathlib
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
NUM_EPOCHS = 200
BATCH_SIZE = 128
BASE_LR = 0.1                  # standard WRN/SGD value at this batch size
WEIGHT_DECAY = 5e-4            # standard WRN value
WARMUP_EPOCHS = 5
MOMENTUM = 0.9

WRN_DEPTH = 28                 # depth = 6n+4
WRN_WIDEN_FACTOR = 4           # try 8 or 10 if you have the compute budget
WRN_DROPOUT = 0.3

MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
AUG_SWITCH_PROB = 0.5          # probability of using cutmix vs mixup per batch
USE_CLASS_WEIGHTS = False   
MAX_CLASS_WEIGHT_RATIO = 5.0
USE_WEIGHTED_SAMPLER = True
CHECKPOINT_SMOOTHING_WINDOW = 3 

EARLY_STOP_PATIENCE = 30
DIAG_EVERY = 5

USE_SWA = True
SWA_START_FRAC = 0.75

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class AddGaussianNoise:
    def __init__(self, std=0.08):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


train_transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

clean_val_transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

robust_val_transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 2.0))], p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    AddGaussianNoise(std=0.08),
])

test_transformer = clean_val_transformer

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parent
train_path = str(BASE_DIR / "train_images")
test_path = str(BASE_DIR / "test_images")
csv_path = str(BASE_DIR / "train_labels.csv")
classes_path = str(BASE_DIR / "classes.txt")

with open(classes_path) as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# ------------------------------------------------------------------
# Datasets
# ------------------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, classes_file, indices=None, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(csv_file, dtype={'id': str})
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
        with open(classes_file) as f:
            classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx]['id'])
        label = self.class_to_idx[self.data.iloc[idx]['label']]
        image = Image.open(os.path.join(self.img_dir, img_id + ".png")).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


full_df = pd.read_csv(csv_path, dtype={'id': str})
n = len(full_df)
indices = np.random.RandomState(SEED).permutation(n)
train_size = int(0.85 * n)
train_indices = indices[:train_size].tolist()
val_indices = indices[train_size:].tolist()

train_dataset = CustomImageDataset(train_path, csv_path, classes_path, train_indices, train_transformer)
clean_val_dataset = CustomImageDataset(train_path, csv_path, classes_path, val_indices, clean_val_transformer)
robust_val_dataset = CustomImageDataset(train_path, csv_path, classes_path, val_indices, robust_val_transformer)
test_dataset = TestDataset(test_path, test_transformer)

train_labels_arr = full_df.iloc[train_indices]['label'].map({c: i for i, c in enumerate(CLASS_NAMES)}).values
class_counts = np.bincount(train_labels_arr, minlength=len(CLASS_NAMES))
print("Train class counts:", dict(zip(CLASS_NAMES, class_counts.tolist())))

if USE_CLASS_WEIGHTS:
    raw_weights = class_counts.sum() / (len(CLASS_NAMES) * np.maximum(class_counts, 1))
    raw_weights = raw_weights / raw_weights.mean()
    raw_weights = np.clip(raw_weights, 1.0 / MAX_CLASS_WEIGHT_RATIO, MAX_CLASS_WEIGHT_RATIO)
    print("Class weights (clamped):", dict(zip(CLASS_NAMES, np.round(raw_weights, 3).tolist())))
    class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(device)
else:
    class_weights = None

if USE_WEIGHTED_SAMPLER:
    per_class_sample_weight = 1.0 / np.maximum(class_counts, 1)
    sample_weights = per_class_sample_weight[train_labels_arr]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    print("Using WeightedRandomSampler for train_loader (rare classes oversampled).")
else:
    sampler = None

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=(sampler is None), sampler=sampler,
    num_workers=2, pin_memory=True, drop_last=True,
)
clean_val_loader = DataLoader(clean_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
robust_val_loader = DataLoader(robust_val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ------------------------------------------------------------------
# Wide ResNet 
# ------------------------------------------------------------------
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=4, num_classes=10, dropout_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must satisfy depth = 6n + 4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.group1 = self._make_group(widths[0], widths[1], n, stride=1, dropout_rate=dropout_rate)
        self.group2 = self._make_group(widths[1], widths[2], n, stride=2, dropout_rate=dropout_rate)
        self.group3 = self._make_group(widths[2], widths[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn_final = nn.BatchNorm2d(widths[3])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_group(self, in_planes, out_planes, n, stride, dropout_rate):
        layers = [WideBasicBlock(in_planes, out_planes, stride, dropout_rate)]
        for _ in range(n - 1):
            layers.append(WideBasicBlock(out_planes, out_planes, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = F.relu(self.bn_final(x))
        x = self.gap(x).view(x.size(0), -1)
        return self.fc(x)


model = WideResNet(depth=WRN_DEPTH, widen_factor=WRN_WIDEN_FACTOR,
                    num_classes=len(CLASS_NAMES), dropout_rate=WRN_DROPOUT).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"WideResNet-{WRN_DEPTH}-{WRN_WIDEN_FACTOR}: {n_params:,} params")


# ------------------------------------------------------------------
# Mixup + CutMix 
# ------------------------------------------------------------------
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    x2, y2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    x1, y1, x2, y2 = rand_bbox(x.size(), lam)
    x[:, :, x1:x2, y1:y2] = x[idx, :, x1:x2, y1:y2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (x.size(-1) * x.size(-2)))
    return x, y, y[idx], lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ------------------------------------------------------------------
# Optimizer / schedule / loss
# ------------------------------------------------------------------
optimizer = SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY, nesterov=True)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)


def set_lr(opt, lr):
    for g in opt.param_groups:
        g['lr'] = lr


if USE_SWA:
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start_epoch = int(NUM_EPOCHS * SWA_START_FRAC)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=BASE_LR * 0.05)

# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
def evaluate(loader, eval_model):
    eval_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = eval_model(images).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, all_labels, all_preds


best_smoothed_f1 = 0.0
epochs_no_improve = 0
robust_f1_history = []  # for checkpoint smoothing — rare classes have ~50-80 val
                         # samples, so single-epoch F1 is noisy; we checkpoint on a
                         # trailing average instead of the raw per-epoch value.

print(f"Train batches: {len(train_loader)} | Clean val: {len(clean_val_loader)} | Robust val: {len(robust_val_loader)}")

for epoch in range(NUM_EPOCHS):
    if epoch < WARMUP_EPOCHS:
        set_lr(optimizer, BASE_LR * (epoch + 1) / WARMUP_EPOCHS)

    model.train()
    train_loss, train_correct, n_seen = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        r = np.random.rand()
        if r < AUG_SWITCH_PROB and CUTMIX_ALPHA > 0:
            mixed_images, y_a, y_b, lam = cutmix_data(images.clone(), labels, CUTMIX_ALPHA)
        elif MIXUP_ALPHA > 0:
            mixed_images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
        else:
            mixed_images, y_a, y_b, lam = images, labels, labels, 1.0

        outputs = model(mixed_images)
        loss = mix_criterion(loss_function, outputs, y_a, y_b, lam)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += torch.sum(preds == labels).item()
        n_seen += images.size(0)

    if USE_SWA and epoch >= swa_start_epoch:
        swa_model.update_parameters(model)
        swa_scheduler.step()
    elif epoch >= WARMUP_EPOCHS:
        cosine_scheduler.step()

    train_loss /= n_seen
    train_accuracy = train_correct / n_seen

    clean_acc, clean_f1, _, _ = evaluate(clean_val_loader, model)
    robust_acc, robust_f1, robust_labels, robust_preds = evaluate(robust_val_loader, model)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] LR: {current_lr:.5f} Loss: {train_loss:.4f} Train Acc: {train_accuracy:.4f} | "
          f"Clean Val Acc: {clean_acc:.4f} F1: {clean_f1:.4f} | Robust Val Acc: {robust_acc:.4f} F1: {robust_f1:.4f}")

    if (epoch + 1) % DIAG_EVERY == 0:
        print("Per-class report (robust val):")
        print(classification_report(robust_labels, robust_preds, target_names=CLASS_NAMES, digits=3, zero_division=0))

    robust_f1_history.append(robust_f1)
    smoothed_f1 = np.mean(robust_f1_history[-CHECKPOINT_SMOOTHING_WINDOW:])

    if smoothed_f1 > best_smoothed_f1:
        best_smoothed_f1 = smoothed_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_model_wrn.pth")
        print(f"Best model saved (smoothed robust F1: {smoothed_f1:.4f})")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break

if USE_SWA:
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
    torch.save(swa_model.module.state_dict(), "swa_model_wrn.pth")
    print("SWA model saved")

print("Training finished. Best smoothed robust-val Macro F1:", best_smoothed_f1)

# ------------------------------------------------------------------
# TTA inference 
# ------------------------------------------------------------------
def tta_predict(eval_model, n_aug=6):
    eval_model.eval()
    tta_transforms = [
        clean_val_transformer,
        transforms.Compose([transforms.Resize((32, 32)), transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
        transforms.Compose([transforms.Resize((32, 32)), transforms.RandomCrop(32, padding=4),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    ]
    while len(tta_transforms) < n_aug:
        tta_transforms.append(tta_transforms[len(tta_transforms) % 3])

    all_logits, img_names_all = None, []
    for t in tta_transforms[:n_aug]:
        ds = TestDataset(test_path, transform=t)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        probs_list, names_list = [], []
        with torch.no_grad():
            for images, names in loader:
                images = images.to(device)
                probs_list.append(F.softmax(eval_model(images), dim=1).cpu().numpy())
                names_list.extend(names)
        view_probs = np.concatenate(probs_list, axis=0)
        img_names_all = names_list
        all_logits = view_probs if all_logits is None else all_logits + view_probs

    all_logits /= n_aug
    return img_names_all, all_logits.argmax(axis=1)


best_model = WideResNet(depth=WRN_DEPTH, widen_factor=WRN_WIDEN_FACTOR,
                         num_classes=len(CLASS_NAMES), dropout_rate=WRN_DROPOUT).to(device)
best_model.load_state_dict(torch.load("best_model_wrn.pth", map_location=device))

names, preds = tta_predict(best_model, n_aug=6)
submission = pd.DataFrame({
    "id": [os.path.splitext(n)[0] for n in names],
    "label": [CLASS_NAMES[p] for p in preds],
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")