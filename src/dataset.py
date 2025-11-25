import torch
from torchvision import transforms, datasets

IMAGE_SIZE = 224

def get_dataloaders(data_dir, batch_size=32):
    train_dir = data_dir + "/train"
    val_dir = data_dir + "/val"

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, train_tf)
    val_ds = datasets.ImageFolder(val_dir, val_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, train_ds.classes
