# train_tumor_clf.py
import os
from pathlib import Path
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# ==========================
#  CONFIG
# ==========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "brain_tumor_dataset"   # glioma/ meningioma/ pituitary/
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL_PATH = MODELS_DIR / "tumor_clf.pt"

BATCH_SIZE = 16
NUM_EPOCHS = 20
VAL_RATIO = 0.2
LR = 1e-4
NUM_WORKERS = 4 if os.name != "nt" else 0  # Windows'da 0 bo'lgani yaxshi


# ==========================
#  DEVICE
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================
#  DATA TRANSFORMS
# ==========================
# ResNet50 standard ImageNet mean/std
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# ==========================
#  DATASET & DATALOADER
# ==========================
def prepare_dataloaders():
    if not DATA_DIR.exists():
        raise RuntimeError(f"DATA_DIR not found: {DATA_DIR}")

    full_dataset = datasets.ImageFolder(
        root=str(DATA_DIR),
        transform=train_transforms,  # train transform, keyin val uchun override qilamiz
    )

    num_samples = len(full_dataset)
    num_val = int(num_samples * VAL_RATIO)
    num_train = num_samples - num_val

    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    # val transformni alohida qo'llaymiz
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    class_names = full_dataset.classes  # folder nomlari bo'yicha: ['glioma', 'meningioma', 'pituitary']
    print("Classes:", class_names)

    return train_loader, val_loader, class_names


# ==========================
#  MODEL
# ==========================
def build_model(num_classes: int):
    # ImageNet bilan pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # oxirgi fully-connected layerni almashtiramiz
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ==========================
#  TRAINING LOOP
# ==========================
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    train_loader, val_loader = dataloaders

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        # Har epoch uchun two phases: train va val
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            start_time = time.time()

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)  # (B, num_classes)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels).item()
                total_samples += batch_size

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            elapsed = time.time() - start_time
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}  (time: {elapsed:.1f}s)")

            # copy best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"*** New best val acc: {best_acc:.4f}")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def main():
    train_loader, val_loader, class_names = prepare_dataloaders()
    num_classes = len(class_names)
    print("Num classes:", num_classes)

    model = build_model(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model = train_model(
        model,
        dataloaders=(train_loader, val_loader),
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
    )

    # class_names ni ham model bilan birga saqlab qo'yamiz (keyin mapping uchun)
    checkpoint = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }

    torch.save(checkpoint, OUTPUT_MODEL_PATH)
    print(f"\nSaved trained model to: {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    main()
