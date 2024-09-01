import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top K classes to display')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction')
    parser.add_argument('--category_names', type=str, help='JSON file mapping category indices to category names')
    return parser.parse_args()

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image

def predict(image_path, model, topk=5):
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0) 
    
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.exp(outputs).topk(topk)
    
    probs = probs.cpu().numpy().squeeze()
    indices = indices.cpu().numpy().squeeze()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    return probs, classes

def main():
    # args = parse_args()
    class Args:
        def __init__(self):
            self.image_path = "C:/Users/Edwar/Documents/projects/traductor/proyecto_udacity/aipnd-project/flowers/test/1/image_06743.jpg"
            self.checkpoint = "model_checkpoint.pth"
            self.top_k = 5
            self.category_names = "cat_to_name.json"
            self.gpu = True

    args = Args()
    
    checkpoint = torch.load(args.checkpoint)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=False)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained=False)
    elif checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=False)
    
    in_features = model.classifier[0].in_features if checkpoint['arch'] in ['vgg16', 'vgg13'] else model.fc.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_units'], len(checkpoint['class_to_idx'])),
        nn.LogSoftmax(dim=1)
    ) if checkpoint['arch'] in ['vgg16', 'vgg13'] else nn.Sequential(
        nn.Linear(in_features, checkpoint['hidden_units']),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(checkpoint['hidden_units'], len(checkpoint['class_to_idx'])),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = {str(i): str(i) for i in range(len(model.class_to_idx))}
    
    probs, classes = predict(args.image_path, model, args.top_k)
    
    print("Probabilities: ", probs)
    print("Classes: ", [cat_to_name[c] for c in classes])

if __name__ == "__main__":
    main()
