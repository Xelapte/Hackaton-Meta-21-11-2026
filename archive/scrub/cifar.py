"""Outils CIFAR pour SCRUB : ResNet-18 adapté au 32x32, données, découpage
forget / retain, entraînement et évaluation par lots."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def resnet18_cifar(num_classes=10):
    """ResNet-18 de torchvision avec une première couche adaptée aux images 32x32."""
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def load_cifar(root, augment=False, fake=False):
    """Renvoie (train, test). fake=True : petites données aléatoires pour tester le code."""
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
    if fake:
        mk = lambda n, s: torchvision.datasets.FakeData(  # noqa: E731
            n, (3, 32, 32), 10, transform=T.Compose(norm), random_offset=s)
        return mk(400, 0), mk(200, 1000)
    tr = torchvision.datasets.CIFAR10(root, train=True, download=False,
                                      transform=T.Compose(aug + norm))
    te = torchvision.datasets.CIFAR10(root, train=False, download=False,
                                      transform=T.Compose(norm))
    return tr, te


def labels_of(ds):
    if hasattr(ds, "targets"):
        return np.array(ds.targets)
    return np.array([int(ds[i][1]) for i in range(len(ds))])


def split_forget(train_ds, mode, target=5, n_forget=100, seed=0):
    """Indices (forget, retain). mode = 'class' (toute la classe) ou 'selective'."""
    y = labels_of(train_ds)
    idx_t = np.flatnonzero(y == target)
    if mode == "class":
        forget = idx_t
    else:
        forget = np.random.RandomState(seed).choice(idx_t, min(n_forget, len(idx_t)),
                                                    replace=False)
    retain = np.setdiff1d(np.arange(len(y)), forget)
    return forget, retain


def loader(ds, idx=None, bs=128, shuffle=False, workers=8):
    if idx is not None:
        ds = torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                       num_workers=workers, pin_memory=True)


def train(model, dl, epochs, lr=0.1, device="cuda", log=print):
    """SGD + momentum + décroissance cosinus : la recette classique pour CIFAR."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    model.to(device)
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=device.startswith("cuda")):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        log(f"époque {ep + 1}/{epochs}  loss {loss.item():.3f}")
    return model


@torch.no_grad()
def predict(model, dl, device="cuda"):
    """Renvoie (loss par exemple, prédiction, étiquette) sur tout le loader."""
    model.eval().to(device)
    losses, preds, ys = [], [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        out = model(x).float()
        losses.append(F.cross_entropy(out, y, reduction="none").cpu())
        preds.append(out.argmax(-1).cpu())
        ys.append(y.cpu())
    return torch.cat(losses).numpy(), torch.cat(preds).numpy(), torch.cat(ys).numpy()


def error(model, dl, device="cuda"):
    _, p, y = predict(model, dl, device)
    return float((p != y).mean())
