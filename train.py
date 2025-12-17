import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import ResNetTemporalAvg
from dataset import ShotDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Configure via env vars so Colab + local both work ---
    data_dir = os.environ.get("DATA_DIR", "data")
    out_path = os.environ.get("OUT_PATH", "shot_model.pth")

    epochs = int(os.environ.get("EPOCHS", "3"))
    batch_size = int(os.environ.get("BATCH_SIZE", "2"))
    lr = float(os.environ.get("LR", "1e-4"))
    val_split = float(os.environ.get("VAL_SPLIT", "0.2"))
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))

    ds = ShotDataset(data_dir)

    # Simple train/val split
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device == "cuda"))

    model = ResNetTemporalAvg(num_classes=2).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for frames, labels in train_loader:
            frames = frames.to(device)      # (B, T, C, H, W)
            labels = labels.to(device)

            opt.zero_grad()
            logits = model(frames)
            loss = crit(logits, labels)
            loss.backward()
            opt.step()

            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                logits = model(frames)
                loss = crit(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_loss /= max(1, len(val_loader))
        acc = correct / max(1, total)

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.3f}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights -> {out_path}")


if __name__ == "__main__":
    main()
