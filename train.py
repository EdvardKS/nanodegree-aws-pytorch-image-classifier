# import subprocess

# def install_package(package_name,quehacer="install"):
#     subprocess.check_call(["pip", quehacer, package_name, "-y"])
# install_package("torch",quehacer="uninstall")
# install_package("torch")


import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on image data')
    parser.add_argument('data_dir', type=str, help='Directory for training, validation, and testing data')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'vgg13', 'resnet18'], help='Model architecture')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the model checkpoint')
    return parser.parse_args()




def main():

    # args = parse_args()
    class Args:
        def __init__(self):
            self.data_dir = "C:/Users/Edwar/Documents/projects/traductor/proyecto_udacity/aipnd-project/flowers"
            self.arch = "vgg16"
            self.hidden_units = 512
            self.learning_rate = 0.001
            self.epochs = 5
            self.gpu = True
            self.save_dir = "model_checkpoint"

    args = Args()

    
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    image_datasets = {
        'train': datasets.ImageFolder(root=args.data_dir + '/train', transform=data_transforms['train']),
        'valid': datasets.ImageFolder(root=args.data_dir + '/valid', transform=data_transforms['valid']),
        'test': datasets.ImageFolder(root=args.data_dir + '/test', transform=data_transforms['test'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=64, shuffle=True),
        'test': DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    }
    
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif args.arch == 'resnet18':
        model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch in ['vgg16', 'vgg13']:
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, len(image_datasets['train'].classes)),
            nn.LogSoftmax(dim=1)
        )
    elif args.arch == 'resnet18':
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, args.hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(args.hidden_units, len(image_datasets['train'].classes)),
            nn.LogSoftmax(dim=1)
        )
    
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if args.arch in ['vgg16', 'vgg13'] else model.fc.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - Training loss: {epoch_loss:.4f}")
        
        model.eval()
        valid_loss = 0
        corrects = 0
        total = 0
        with torch.no_grad():
            for images, labels in dataloaders['valid']:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels).item()
                total += labels.size(0)
        
        valid_loss /= len(dataloaders['valid'].dataset)
        valid_acc = corrects / total
        print(f"Epoch {epoch+1}/{args.epochs} - Validation loss: {valid_loss:.4f}, Validation accuracy: {valid_acc:.4f}")
    
    checkpoint = {
        'arch': args.arch,
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx
    }
    torch.save(checkpoint, f"{args.save_dir}/model_checkpoint.pth")
    print(f"Model saved to {args.save_dir}/model_checkpoint.pth")

if __name__ == "__main__":
    main()
