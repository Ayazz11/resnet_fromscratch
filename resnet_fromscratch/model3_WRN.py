
import os
import copy
import random
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split


# ================================================================
# 1. REPRODUCIBILITY
# ================================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Optional deterministic behavior.
# Turn this OFF if maximum training speed is more important.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# ================================================================
# 2. DEVICE
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ================================================================
# 3. CONFIGURATION
# ================================================================

NUM_EPOCHS = 200

BATCH_SIZE = 128

NUM_WORKERS = 0

BASE_LR = 0.1

WEIGHT_DECAY = 5e-4

MOMENTUM = 0.9

WARMUP_EPOCHS = 5


# ---------------- MODEL ----------------

WRN_DEPTH = 40
WRN_WIDEN_FACTOR = 8
WRN_DROPOUT = 0.2


# ---------------- AUGMENTATION ----------------

MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0

# Probability that a batch uses CutMix.
CUTMIX_PROB = 0.35

# Probability that a batch uses MixUp.
MIXUP_PROB = 0.35


# ---------------- CLASS BALANCING ----------------

USE_WEIGHTED_SAMPLER = True

SAMPLER_POWER = 0.5


# ---------------- LOSS ----------------

LABEL_SMOOTHING = 0.05

USE_FOCAL = False

FOCAL_GAMMA = 1.5


# ---------------- EMA ----------------

USE_EMA = True

EMA_DECAY = 0.999


# ---------------- VALIDATION ----------------

ROBUST_VAL_REPEATS = 3

CHECKPOINT_SMOOTHING_WINDOW = 5

EARLY_STOP_PATIENCE = 60

MIN_EPOCHS_BEFORE_EARLY_STOP = 100

DIAG_EVERY = 5


# ---------------- TTA ----------------

USE_TTA = True

TTA_VIEWS = 8


# ---------------- AMP ----------------

USE_AMP = True


# ================================================================
# 4. PATHS
# ================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

train_path = os.path.join(SCRIPT_DIR, "train_images")
test_path = os.path.join(SCRIPT_DIR, "test_images")
csv_path = os.path.join(SCRIPT_DIR, "train_labels.csv")
classes_path = os.path.join(SCRIPT_DIR, "classes.txt")


# ================================================================
# 5. CLASS NAMES
# ================================================================

with open(classes_path, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

NUM_CLASSES = len(CLASS_NAMES)

print("\nClasses:")
for i, c in enumerate(CLASS_NAMES):
    print(i, c)


# ================================================================
# 6. NORMALIZATION
# ================================================================

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


# ================================================================
# 7. CUSTOM GAUSSIAN NOISE
# ================================================================

class AddGaussianNoise:
    def __init__(self, std=0.06):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std


# ================================================================
# 8. TRAINING TRANSFORMS
# ================================================================

standard_train_transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandAugment(
        num_ops=2,
        magnitude=7
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    ),

    transforms.RandomErasing(
        p=0.20,
        scale=(0.02, 0.12)
    )
])


robust_train_transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandAugment(
        num_ops=1,
        magnitude=5
    ),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.5, 1.5)
            )
        ],
        p=0.35
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.20
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    ),

    AddGaussianNoise(
        std=0.06
    ),

    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.10)
    )
])


# ================================================================
# 9. VALIDATION TRANSFORMS
# ================================================================

clean_val_transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )
])


robust_val_transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.RandomApply(
        [
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.5, 2.0)
            )
        ],
        p=0.5
    ),

    transforms.ColorJitter(
        brightness=0.30,
        contrast=0.30,
        saturation=0.30
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    ),

    AddGaussianNoise(
        std=0.08
    )
])


# ================================================================
# 10. DATASET
# ================================================================

class CustomImageDataset(Dataset):

    def __init__(
        self,
        img_dir,
        csv_file,
        classes_file,
        indices=None,
        transform=None
    ):

        self.img_dir = img_dir

        self.transform = transform

        self.data = pd.read_csv(
            csv_file,
            dtype={"id": str}
        )

        if indices is not None:
            self.data = (
                self.data
                .iloc[indices]
                .reset_index(drop=True)
            )

        with open(classes_file, "r") as f:
            classes = [
                line.strip()
                for line in f.readlines()
            ]

        self.class_to_idx = {
            c: i
            for i, c in enumerate(classes)
        }


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):

        img_id = str(
            self.data.iloc[idx]["id"]
        )

        label_name = self.data.iloc[idx]["label"]

        label = self.class_to_idx[label_name]

        image_path = os.path.join(
            self.img_dir,
            img_id + ".png"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ================================================================
# 11. TEST DATASET
# ================================================================

class TestDataset(Dataset):

    def __init__(
        self,
        img_dir,
        transform=None
    ):

        self.img_dir = img_dir

        self.transform = transform

        self.images = sorted(
            os.listdir(img_dir)
        )


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_name = self.images[idx]

        image_path = os.path.join(
            self.img_dir,
            img_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, img_name


# ================================================================
# 12. LOAD LABEL DATA
# ================================================================

full_df = pd.read_csv(
    csv_path,
    dtype={"id": str}
)

n = len(full_df)

all_indices = np.arange(n)

label_to_idx = {
    c: i
    for i, c in enumerate(CLASS_NAMES)
}

all_labels = (
    full_df["label"]
    .map(label_to_idx)
    .values
)


# ================================================================
# 13. STRATIFIED TRAIN / VALIDATION SPLIT
# ================================================================

train_indices, val_indices = train_test_split(

    all_indices,

    test_size=0.15,

    stratify=all_labels,

    random_state=SEED
)

train_indices = train_indices.tolist()
val_indices = val_indices.tolist()


print("\nDataset size:", n)
print("Training:", len(train_indices))
print("Validation:", len(val_indices))


# ================================================================
# 14. DATASETS
# ================================================================

train_dataset = CustomImageDataset(

    train_path,

    csv_path,

    classes_path,

    train_indices,

    standard_train_transform
)


robust_train_dataset = CustomImageDataset(

    train_path,

    csv_path,

    classes_path,

    train_indices,

    robust_train_transform
)


clean_val_dataset = CustomImageDataset(

    train_path,

    csv_path,

    classes_path,

    val_indices,

    clean_val_transform
)


robust_val_dataset = CustomImageDataset(

    train_path,

    csv_path,

    classes_path,

    val_indices,

    robust_val_transform
)


# ================================================================
# 15. CLASS COUNTS
# ================================================================

train_labels_arr = (

    full_df

    .iloc[train_indices]

    ["label"]

    .map(label_to_idx)

    .values
)


class_counts = np.bincount(

    train_labels_arr,

    minlength=NUM_CLASSES
)


print("\nTraining class counts:")

for c, count in zip(
    CLASS_NAMES,
    class_counts
):

    print(
        f"{c:15s}: {count}"
    )


# ================================================================
# 16. SOFT CLASS BALANCING
# ================================================================

if USE_WEIGHTED_SAMPLER:

    per_class_weight = (

        1.0 /
        np.power(
            np.maximum(class_counts, 1),
            SAMPLER_POWER
        )

    )

    sample_weights = (
        per_class_weight[
            train_labels_arr
        ]
    )

    sampler = WeightedRandomSampler(

        weights=torch.tensor(
            sample_weights,
            dtype=torch.double
        ),

        num_samples=len(
            sample_weights
        ),

        replacement=True
    )

    print(
        "\nUsing sqrt-inverse WeightedRandomSampler."
    )

else:

    sampler = None

    print(
        "\nUsing normal shuffled sampling."
    )


# ================================================================
# 17. TRAINING LOADER
# ================================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=(
        sampler is None
    ),

    sampler=sampler,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    drop_last=True
)


robust_train_loader = DataLoader(

    robust_train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=(
        sampler is None
    ),

    sampler=sampler,

    num_workers=NUM_WORKERS,

    pin_memory=True,

    drop_last=True
)


clean_val_loader = DataLoader(

    clean_val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)


# ================================================================
# 18. FIXED ROBUST VALIDATION
# ================================================================

print(
    "\nMaterializing fixed robust validation set..."
)


robust_val_loader_temp = DataLoader(

    robust_val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS
)


robust_images = []
robust_labels = []


for images, labels in robust_val_loader_temp:

    robust_images.append(
        images
    )

    robust_labels.append(
        labels
    )


robust_images = torch.cat(
    robust_images,
    dim=0
)

robust_labels = torch.cat(
    robust_labels,
    dim=0
)


robust_val_fixed_dataset = torch.utils.data.TensorDataset(

    robust_images,

    robust_labels
)


robust_val_loader = DataLoader(

    robust_val_fixed_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)


print(
    "Fixed robust validation:",
    len(robust_val_fixed_dataset)
)


# ================================================================
# 19. WIDE RESNET
# ================================================================

class WideBasicBlock(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        stride,
        dropout_rate
    ):

        super().__init__()

        self.bn1 = nn.BatchNorm2d(
            in_planes
        )

        self.conv1 = nn.Conv2d(

            in_planes,

            out_planes,

            kernel_size=3,

            stride=stride,

            padding=1,

            bias=False
        )

        self.dropout = nn.Dropout(
            p=dropout_rate
        )

        self.bn2 = nn.BatchNorm2d(
            out_planes
        )

        self.conv2 = nn.Conv2d(

            out_planes,

            out_planes,

            kernel_size=3,

            stride=1,

            padding=1,

            bias=False
        )

        self.shortcut = None

        if (
            stride != 1
            or in_planes != out_planes
        ):

            self.shortcut = nn.Conv2d(

                in_planes,

                out_planes,

                kernel_size=1,

                stride=stride,

                bias=False
            )


    def forward(self, x):

        out = F.relu(
            self.bn1(x)
        )

        if self.shortcut is not None:

            shortcut = self.shortcut(
                out
            )

        else:

            shortcut = x

        out = self.conv1(
            out
        )

        out = self.dropout(
            out
        )

        out = self.conv2(
            F.relu(
                self.bn2(out)
            )
        )

        return out + shortcut


# ================================================================
# 20. WIDE RESNET MODEL
# ================================================================

class WideResNet(nn.Module):

    def __init__(

        self,

        depth=40,

        widen_factor=8,

        num_classes=10,

        dropout_rate=0.2

    ):

        super().__init__()

        assert (
            depth - 4
        ) % 6 == 0

        n = (
            depth - 4
        ) // 6

        widths = [

            16,

            16 * widen_factor,

            32 * widen_factor,

            64 * widen_factor

        ]

        self.conv1 = nn.Conv2d(

            3,

            widths[0],

            kernel_size=3,

            stride=1,

            padding=1,

            bias=False
        )

        self.group1 = self._make_group(

            widths[0],

            widths[1],

            n,

            stride=1,

            dropout_rate=dropout_rate
        )

        self.group2 = self._make_group(

            widths[1],

            widths[2],

            n,

            stride=2,

            dropout_rate=dropout_rate
        )

        self.group3 = self._make_group(

            widths[2],

            widths[3],

            n,

            stride=2,

            dropout_rate=dropout_rate
        )

        self.bn_final = nn.BatchNorm2d(

            widths[3]
        )

        self.gap = nn.AdaptiveAvgPool2d(

            (1, 1)
        )

        self.fc = nn.Linear(

            widths[3],

            num_classes
        )

        self._initialize_weights()


    def _make_group(

        self,

        in_planes,

        out_planes,

        n,

        stride,

        dropout_rate

    ):

        layers = [

            WideBasicBlock(

                in_planes,

                out_planes,

                stride,

                dropout_rate

            )
        ]

        for _ in range(n - 1):

            layers.append(

                WideBasicBlock(

                    out_planes,

                    out_planes,

                    1,

                    dropout_rate

                )
            )

        return nn.Sequential(*layers)


    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(
                m,
                nn.Conv2d
            ):

                nn.init.kaiming_normal_(

                    m.weight,

                    mode="fan_out",

                    nonlinearity="relu"
                )

            elif isinstance(
                m,
                nn.BatchNorm2d
            ):

                nn.init.constant_(
                    m.weight,
                    1
                )

                nn.init.constant_(
                    m.bias,
                    0
                )

            elif isinstance(
                m,
                nn.Linear
            ):

                nn.init.normal_(
                    m.weight,
                    0,
                    0.01
                )

                nn.init.constant_(
                    m.bias,
                    0
                )


    def forward_features(self, x):

        x = self.conv1(x)

        x = self.group1(x)

        x = self.group2(x)

        x = self.group3(x)

        x = F.relu(
            self.bn_final(x)
        )

        x = self.gap(x)

        x = x.view(
            x.size(0),
            -1
        )

        return x


    def forward(self, x):

        x = self.forward_features(x)

        return self.fc(x)


# ================================================================
# 21. CREATE MODEL
# ================================================================

model = WideResNet(

    depth=WRN_DEPTH,

    widen_factor=WRN_WIDEN_FACTOR,

    num_classes=NUM_CLASSES,

    dropout_rate=WRN_DROPOUT

).to(device)


num_params = sum(

    p.numel()

    for p in model.parameters()

)


print(
    f"\nWRN-{WRN_DEPTH}-{WRN_WIDEN_FACTOR}"
)

print(
    f"Parameters: {num_params:,}"
)


# ================================================================
# 22. MIXUP
# ================================================================

def mixup_data(

    x,

    y,

    alpha

):

    if alpha <= 0:

        return (
            x,
            y,
            y,
            1.0
        )

    lam = np.random.beta(
        alpha,
        alpha
    )

    idx = torch.randperm(

        x.size(0),

        device=x.device
    )

    mixed_x = (

        lam * x

        +

        (1 - lam)
        * x[idx]

    )

    return (

        mixed_x,

        y,

        y[idx],

        lam

    )


# ================================================================
# 23. CUTMIX
# ================================================================

def rand_bbox(

    size,

    lam

):

    W = size[2]

    H = size[3]

    cut_rat = np.sqrt(
        1.0 - lam
    )

    cut_w = int(
        W * cut_rat
    )

    cut_h = int(
        H * cut_rat
    )

    cx = np.random.randint(W)

    cy = np.random.randint(H)

    x1 = np.clip(
        cx - cut_w // 2,
        0,
        W
    )

    y1 = np.clip(
        cy - cut_h // 2,
        0,
        H
    )

    x2 = np.clip(
        cx + cut_w // 2,
        0,
        W
    )

    y2 = np.clip(
        cy + cut_h // 2,
        0,
        H
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


def cutmix_data(

    x,

    y,

    alpha

):

    if alpha <= 0:

        return (
            x,
            y,
            y,
            1.0
        )

    lam = np.random.beta(
        alpha,
        alpha
    )

    idx = torch.randperm(

        x.size(0),

        device=x.device
    )

    x1, y1, x2, y2 = rand_bbox(

        x.size(),

        lam
    )

    x[:, :, x1:x2, y1:y2] = (

        x[
            idx,
            :,
            x1:x2,
            y1:y2
        ]

    )

    lam = (

        1 -

        (
            (x2 - x1)
            *
            (y2 - y1)
        )

        /

        (
            x.size(-1)
            *
            x.size(-2)
        )
    )

    return (

        x,

        y,

        y[idx],

        lam

    )


# ================================================================
# 24. LOSS
# ================================================================

class FocalLoss(nn.Module):

    def __init__(

        self,

        gamma=1.5,

        label_smoothing=0.05

    ):

        super().__init__()

        self.gamma = gamma

        self.label_smoothing = (
            label_smoothing
        )


    def forward(

        self,

        logits,

        targets

    ):

        ce = F.cross_entropy(

            logits,

            targets,

            reduction="none",

            label_smoothing=self.label_smoothing

        )

        pt = torch.exp(-ce)

        loss = (

            (1 - pt)
            ** self.gamma
            * ce
        )

        return loss.mean()


if USE_FOCAL:

    loss_function = FocalLoss(

        gamma=FOCAL_GAMMA,

        label_smoothing=LABEL_SMOOTHING

    )

else:

    loss_function = nn.CrossEntropyLoss(

        label_smoothing=LABEL_SMOOTHING

    )


# ================================================================
# 25. OPTIMIZER
# ================================================================

optimizer = SGD(

    model.parameters(),

    lr=BASE_LR,

    momentum=MOMENTUM,

    weight_decay=WEIGHT_DECAY,

    nesterov=True

)


# ================================================================
# 26. LR SCHEDULER
# ================================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=NUM_EPOCHS - WARMUP_EPOCHS,

    eta_min=BASE_LR * 0.01

)


# ================================================================
# 27. EMA
# ================================================================

if USE_EMA:

    ema_model = copy.deepcopy(
        model
    )

    ema_model.eval()

    for p in ema_model.parameters():

        p.requires_grad = False

else:

    ema_model = None


@torch.no_grad()
def update_ema(

    ema_model,

    model,

    decay

):

    ema_params = dict(
        ema_model.named_parameters()
    )

    model_params = dict(
        model.named_parameters()
    )

    for name in ema_params:

        ema_params[name].mul_(decay)

        ema_params[name].add_(

            model_params[name],

            alpha=1.0 - decay

        )

    ema_buffers = dict(
        ema_model.named_buffers()
    )

    model_buffers = dict(
        model.named_buffers()
    )

    for name in ema_buffers:

        ema_buffers[name].copy_(
            model_buffers[name]
        )


# ================================================================
# 28. AMP
# ================================================================

amp_scaler = torch.amp.GradScaler(

    "cuda",

    enabled=(
        USE_AMP
        and device.type == "cuda"
    )

)


# ================================================================
# 29. EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(

    loader,

    eval_model

):

    eval_model.eval()

    all_preds = []

    all_labels = []

    all_logits = []

    for images, labels in tqdm(loader):

        images = images.to(
            device,
            non_blocking=True
        )

        logits = eval_model(
            images
        )

        preds = logits.argmax(
            dim=1
        )

        all_logits.append(
            logits.cpu()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

    all_logits = torch.cat(
        all_logits,
        dim=0
    ).numpy()

    all_labels = np.array(
        all_labels
    )

    all_preds = np.array(
        all_preds
    )

    accuracy = (
        all_preds == all_labels
    ).mean()

    macro_f1 = f1_score(

        all_labels,

        all_preds,

        average="macro"

    )

    return (

        accuracy,

        macro_f1,

        all_labels,

        all_preds,

        all_logits

    )


# ================================================================
# 30. TRAINING BATCH MIXING
# ================================================================

def create_mixed_batch(

    images,

    labels

):

    r = np.random.rand()

    if (

        r < CUTMIX_PROB

        and CUTMIX_ALPHA > 0

    ):

        return cutmix_data(

            images.clone(),

            labels,

            CUTMIX_ALPHA

        )

    elif (

        r < (
            CUTMIX_PROB
            +
            MIXUP_PROB
        )

        and MIXUP_ALPHA > 0

    ):

        return mixup_data(

            images,

            labels,

            MIXUP_ALPHA

        )

    else:

        return (

            images,

            labels,

            labels,

            1.0

        )


# ================================================================
# 31. MIXED LOSS
# ================================================================

def mixed_loss(

    criterion,

    predictions,

    y_a,

    y_b,

    lam

):

    return (

        lam * criterion(
            predictions,
            y_a
        )

        +

        (1 - lam)
        *
        criterion(
            predictions,
            y_b
        )

    )


# ================================================================
# 32. TRAINING
# ================================================================

best_robust_f1 = 0.0

best_clean_f1 = 0.0

epochs_without_improvement = 0

robust_f1_history = []


print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):


    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    if epoch < WARMUP_EPOCHS:

        lr = (

            BASE_LR
            *
            (epoch + 1)
            /
            WARMUP_EPOCHS

        )

        for param_group in optimizer.param_groups:

            param_group["lr"] = lr


    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    samples_seen = 0


    # We alternate between normal and robustness-oriented datasets.
    #
    # This means the network is not trained exclusively on corrupted
    # images.

    use_robust_loader = (
        epoch % 2 == 1
    )

    current_loader = (

        robust_train_loader

        if use_robust_loader

        else train_loader

    )


    for images, labels in tqdm(current_loader):

        images = images.to(

            device,

            non_blocking=True

        )

        labels = labels.to(

            device,

            non_blocking=True

        )


        optimizer.zero_grad(
            set_to_none=True
        )


        mixed_images, y_a, y_b, lam = (

            create_mixed_batch(
                images,
                labels
            )

        )


        with torch.amp.autocast(

            "cuda",

            enabled=(
                USE_AMP
                and device.type == "cuda"
            )

        ):

            outputs = model(
                mixed_images
            )

            loss = mixed_loss(

                loss_function,

                outputs,

                y_a,

                y_b,

                lam

            )


        amp_scaler.scale(
            loss
        ).backward()


        amp_scaler.unscale_(
            optimizer
        )


        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=5.0

        )


        amp_scaler.step(
            optimizer
        )

        amp_scaler.update()


        if USE_EMA:

            update_ema(

                ema_model,

                model,

                EMA_DECAY

            )


        running_loss += (

            loss.item()
            *
            images.size(0)

        )


        preds = outputs.argmax(
            dim=1
        )

        correct += (

            (preds == labels)
            .sum()
            .item()

        )

        samples_seen += images.size(0)


    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------

    if epoch >= WARMUP_EPOCHS:

        scheduler.step()


    train_loss = (

        running_loss
        /
        samples_seen

    )

    train_accuracy = (

        correct
        /
        samples_seen

    )


    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    clean_acc, clean_f1, clean_labels, clean_preds, clean_logits = (

        evaluate(

            clean_val_loader,

            model

        )

    )


    robust_acc, robust_f1, robust_labels, robust_preds, robust_logits = (

        evaluate(

            robust_val_loader,

            model

        )

    )


    # EMA evaluation
    if USE_EMA:

        ema_clean_acc, ema_clean_f1, _, _, _ = (

            evaluate(

                clean_val_loader,

                ema_model

            )

        )

        ema_robust_acc, ema_robust_f1, _, _, _ = (

            evaluate(

                robust_val_loader,

                ema_model

            )

        )

    else:

        ema_clean_f1 = clean_f1

        ema_robust_f1 = robust_f1


    current_lr = optimizer.param_groups[0]["lr"]


    print(

        f"\nEpoch [{epoch+1:03d}/{NUM_EPOCHS}] "

        f"LR={current_lr:.6f} "

        f"Loss={train_loss:.4f} "

        f"TrainAcc={train_accuracy:.4f}"

    )

    print(

        f"Model  | "

        f"Clean Acc={clean_acc:.4f} "

        f"Clean F1={clean_f1:.4f} "

        f"Robust Acc={robust_acc:.4f} "

        f"Robust F1={robust_f1:.4f}"

    )

    print(

        f"EMA    | "

        f"Clean F1={ema_clean_f1:.4f} "

        f"Robust F1={ema_robust_f1:.4f}"

    )


    # ------------------------------------------------------------
    # Per-class diagnostics
    # ------------------------------------------------------------

    if (

        (epoch + 1) % DIAG_EVERY == 0

    ):

        print("\nRobust validation classification report:")

        print(

            classification_report(

                robust_labels,

                robust_preds,

                target_names=CLASS_NAMES,

                digits=3,

                zero_division=0

            )

        )


    # ------------------------------------------------------------
    # Checkpoint based on EMA robust F1
    # ------------------------------------------------------------

    robust_f1_history.append(
        ema_robust_f1
    )

    smoothed_f1 = np.mean(

        robust_f1_history[
            -CHECKPOINT_SMOOTHING_WINDOW:
        ]

    )


    if smoothed_f1 > best_robust_f1:

        best_robust_f1 = smoothed_f1

        best_clean_f1 = ema_clean_f1

        epochs_without_improvement = 0


        torch.save(

            ema_model.state_dict()
            if USE_EMA
            else model.state_dict(),

            "best_model.pth"

        )


        print(

            f">>> BEST MODEL SAVED | "
            f"Smoothed Robust F1 = "
            f"{smoothed_f1:.4f}"

        )

    else:

        epochs_without_improvement += 1


    # ------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------

    if (

        epoch + 1
        >= MIN_EPOCHS_BEFORE_EARLY_STOP

        and

        epochs_without_improvement
        >= EARLY_STOP_PATIENCE

    ):

        print(
            "\nEarly stopping."
        )

        break


# ================================================================
# 33. LOAD BEST MODEL
# ================================================================

print("\n")
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


best_model = WideResNet(

    depth=WRN_DEPTH,

    widen_factor=WRN_WIDEN_FACTOR,

    num_classes=NUM_CLASSES,

    dropout_rate=WRN_DROPOUT

).to(device)


best_model.load_state_dict(

    torch.load(

        "best_model.pth",

        map_location=device

    )

)


best_model.eval()


# ================================================================
# 34. FINAL VALIDATION
# ================================================================

final_clean_acc, final_clean_f1, clean_labels, clean_preds, clean_logits = (

    evaluate(

        clean_val_loader,

        best_model

    )

)


final_robust_acc, final_robust_f1, robust_labels, robust_preds, robust_logits = (

    evaluate(

        robust_val_loader,

        best_model

    )

)


print("\n")
print("=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

print(
    f"Clean Accuracy : {final_clean_acc:.4f}"
)

print(
    f"Clean Macro F1 : {final_clean_f1:.4f}"
)

print(
    f"Robust Accuracy: {final_robust_acc:.4f}"
)

print(
    f"Robust Macro F1: {final_robust_f1:.4f}"
)


print("\nFinal robust classification report:")

print(

    classification_report(

        robust_labels,

        robust_preds,

        target_names=CLASS_NAMES,

        digits=4,

        zero_division=0

    )

)


# ================================================================
# 35. CLASS-BIAS CALIBRATION
# ================================================================

def calibrated_predictions(

    logits,

    bias

):

    return np.argmax(

        logits + bias[None, :],

        axis=1

    )


baseline_bias = np.zeros(
    NUM_CLASSES,
    dtype=np.float32
)


baseline_f1 = f1_score(

    robust_labels,

    calibrated_predictions(
        robust_logits,
        baseline_bias
    ),

    average="macro"

)


print("\n")
print("=" * 70)
print("F1 CALIBRATION")
print("=" * 70)

print(
    "Baseline robust F1:",
    round(baseline_f1, 5)
)


# ------------------------------------------------------------
# Coordinate search
# ------------------------------------------------------------

best_bias = np.zeros(
    NUM_CLASSES,
    dtype=np.float32
)

best_f1 = baseline_f1


# Search range.
#
# Small values are intentional.
# We don't want calibration to completely distort the classifier.

SEARCH_VALUES = np.array([

    -0.30,
    -0.20,
    -0.10,
     0.00,
     0.10,
     0.20,
     0.30

])


# Coordinate descent.
#
# We optimize one class at a time.

for iteration in range(3):

    improved = False

    for c in range(NUM_CLASSES):

        current_best = best_bias[c]

        local_best_f1 = best_f1


        for value in SEARCH_VALUES:

            candidate_bias = (
                best_bias.copy()
            )

            candidate_bias[c] = value


            preds = calibrated_predictions(

                robust_logits,

                candidate_bias

            )


            score = f1_score(

                robust_labels,

                preds,

                average="macro"

            )


            if score > local_best_f1:

                local_best_f1 = score

                current_best = value


        if current_best != best_bias[c]:

            best_bias[c] = current_best

            best_f1 = local_best_f1

            improved = True


    if not improved:

        break


print("\nClass biases:")

for name, bias in zip(
    CLASS_NAMES,
    best_bias
):

    print(
        f"{name:15s}: {bias:+.3f}"
    )


print(
    "\nCalibrated robust F1:",
    round(best_f1, 5)
)


# Save calibration.
np.save(
    "class_bias.npy",
    best_bias
)


# ================================================================
# 36. TEST DATASET
# ================================================================

test_base_transform = transforms.Compose([

    transforms.Resize((32, 32)),

    transforms.ToTensor(),

    transforms.Normalize(
        MEAN,
        STD
    )

])


test_dataset = TestDataset(

    test_path,

    test_base_transform

)


# ================================================================
# 37. TTA TRANSFORMS
# ================================================================

tta_transforms = []


# ------------------------------------------------------------
# 1. Original
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((32, 32)),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ------------------------------------------------------------
# 2. Horizontal flip
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((32, 32)),

        transforms.RandomHorizontalFlip(
            p=1.0
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ------------------------------------------------------------
# 3. Center crop after resize
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((36, 36)),

        transforms.CenterCrop(32),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ------------------------------------------------------------
# 4. Mild brightness
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((32, 32)),

        transforms.ColorJitter(
            brightness=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ------------------------------------------------------------
# 5. Mild contrast
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((32, 32)),

        transforms.ColorJitter(
            contrast=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ------------------------------------------------------------
# 6. Mild saturation
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((32, 32)),

        transforms.ColorJitter(
            saturation=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ------------------------------------------------------------
# 7. Mild blur
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((32, 32)),

        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.5, 1.0)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ------------------------------------------------------------
# 8. Crop + flip
# ------------------------------------------------------------

tta_transforms.append(

    transforms.Compose([

        transforms.Resize((36, 36)),

        transforms.RandomCrop(32),

        transforms.RandomHorizontalFlip(
            p=1.0
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            MEAN,
            STD
        )

    ])

)


# ================================================================
# 38. TTA PREDICTION
# ================================================================

@torch.no_grad()
def tta_predict(

    eval_model,

    transforms_list

):

    eval_model.eval()

    accumulated_probs = None

    image_names = None


    for view_idx, transform in enumerate(

        transforms_list

    ):

        print(

            f"TTA view "
            f"{view_idx + 1}/"
            f"{len(transforms_list)}"

        )


        dataset = TestDataset(

            test_path,

            transform

        )


        loader = DataLoader(

            dataset,

            batch_size=BATCH_SIZE,

            shuffle=False,

            num_workers=NUM_WORKERS,

            pin_memory=True

        )


        view_probs = []

        names = []


        for images, batch_names in loader:

            images = images.to(

                device,

                non_blocking=True

            )

            logits = eval_model(
                images
            )

            probs = F.softmax(

                logits,

                dim=1

            )

            view_probs.append(
                probs.cpu().numpy()
            )

            names.extend(
                batch_names
            )


        view_probs = np.concatenate(

            view_probs,

            axis=0

        )


        if accumulated_probs is None:

            accumulated_probs = (
                view_probs
            )

            image_names = names

        else:

            accumulated_probs += (
                view_probs
            )


    accumulated_probs /= len(
        transforms_list
    )


    return (

        image_names,

        accumulated_probs

    )


# ================================================================
# 39. FINAL TEST PREDICTION
# ================================================================

print("\n")
print("=" * 70)
print("FINAL TEST INFERENCE")
print("=" * 70)


if USE_TTA:

    test_names, test_probs = tta_predict(

        best_model,

        tta_transforms[:TTA_VIEWS]

    )

else:

    test_names, test_probs = tta_predict(

        best_model,

        tta_transforms[:1]

    )


# ================================================================
# 40. APPLY F1 CALIBRATION
# ================================================================

test_logits_for_calibration = np.log(

    np.clip(

        test_probs,

        1e-8,

        1.0

    )

)


test_preds = np.argmax(

    test_logits_for_calibration

    +

    best_bias[None, :],

    axis=1

)


# ================================================================
# 41. SUBMISSION
# ================================================================

submission = pd.DataFrame({

    "id": [

        os.path.splitext(name)[0]

        for name in test_names

    ],

    "label": [

        CLASS_NAMES[p]

        for p in test_preds

    ]

})


submission_path = "submission.csv"


submission.to_csv(

    submission_path,

    index=False

)


print("\n")
print("=" * 70)
print("SUBMISSION CREATED")
print("=" * 70)

print(
    submission.head(10)
)

print(
    "\nSubmission shape:",
    submission.shape
)

print(
    "\nPrediction distribution:"
)

print(
    submission["label"]
    .value_counts()
)


print(
    f"\nSaved to: {submission_path}"
)