# Deep learning
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder

# Custom modules
from src.dataset import DatasetCreator

# Data processing
from PIL import Image
import numpy as np
import urllib.error
import socket
from tqdm import tqdm
import copy
import os


class CNN_ResNet18(nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    def __init__(self, n_classes=2, gray=False):
        super(CNN_ResNet18, self).__init__()

        self.n_classes       = n_classes
        self.gray            = gray
        self.out_layer       = nn.Linear(512, self.n_classes)
        self.train_dataset   = None
        self.val_dataset     = None

        # Model
        self.model = None
        self._init_model_architecture()
        self._init_hyperparameters()
        self._init_transforms()

    def _init_model_architecture(self):
        try:
            # Load pre-trained model
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        except (urllib.error.URLError, urllib.error.HTTPError, socket.gaierror, RuntimeError) as e:
            # Load no pre-trained model
            link = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
            path = r"C:\Users\your_user\.cache\torch\hub\checkpoints\\"
            print("Error downloading model:", e)
            print(f"Try downloading the model manually from {link} and move it to {path} and try again.")
            print("Loading model without pre-trained weights...")
            self.model = models.resnet18(weights=None)

        finally:
            # Set model as feature extractor
            print("Setting model as features extractor...")
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            # Replace output layer
            print("Replacing output layer...")
            self.model.fc = self.out_layer
            # Set device
            print(f"Moving model to {self.device}...")
            self.model.to(self.device)
            print("Successfully created model.")

    def _init_hyperparameters(self):
        self.n_epochs       = 10
        self.batch_size     = 32
        self.lr             = 0.000001
        self.momentum       = 0.9
        self.lr_scheduler   = True
        self.base_lr        = 0.000001
        self.max_lr         = 0.001
        self.scheduler      = None
        self.optimizer      = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        if self.lr_scheduler:
            # Create scheduler
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.base_lr,   # Minimum learning rate
                max_lr=self.max_lr,     # Maximum learning rate
                step_size_up=5,         # Steps to increase LR from base_lr to max_lr
                mode="triangular2"      # Learning rate follows triangular cycle and halves the max_lr each cycle
            )

    def _init_transforms(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            #transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if self.gray:
            self.train_transform.transforms.insert(1, transforms.Grayscale(num_output_channels=3))
            self.val_transform.transforms.insert(1, transforms.Grayscale(num_output_channels=3))

    def create_dataset(self, root, output_dir, train_ratio=0.9):
        # Create instance
        dataset = DatasetCreator(output_dir=output_dir, train_ratio=train_ratio)
        # Create dataset
        self.train_dataset, self.val_dataset = dataset.create_dataset(root, self.train_transform, self.val_transform)
        # Update N classes and output layer
        self.update_n_classes(self.train_dataset.classes)
        # Free memory
        del dataset

    def update_n_classes(self, classes_list):
        # Update attribute
        self.n_classes = len(classes_list)
        # Update output layer
        self.out_layer = nn.Linear(512, self.n_classes)
        self.model.fc = self.out_layer
        print(f"{self.n_classes} model classes: {classes_list}")

    def update_hypams(self, n_epochs=10, batch_size=32, lr=0.000001, momentum=0.9, base_lr=0.000001, max_lr=0.001):
        # Store new parameters
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.lr             = lr
        self.momentum       = momentum
        self.base_lr        = base_lr
        self.max_lr         = max_lr

        # Update optimizer as it depends on the learning rate
        self.optimizer      = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # Update lr scheduler
        if self.lr_scheduler:
            # Create scheduler
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.base_lr,   # Minimum learning rate
                max_lr=self.max_lr,     # Maximum learning rate
                step_size_up=5,         # Steps to increase LR from base_lr to max_lr
                mode="triangular2"      # Learning rate follows triangular cycle and halves the max_lr each cycle
            )

    def _train(self, print_=True):        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=1)

        loss_list = []  # Store average training loss per epoch
        accuracy_list = []  # Store validation accuracy per epoch
        correct = 0

        n_test = len(self.val_dataset)  # Total number of validation samples
        accuracy_best = 0  # Track best validation accuracy
        best_model_wts = copy.deepcopy(self.model.state_dict())  # Backup best model weights

        print("Starting training. This might take a few minutes...")

        for epoch in tqdm(range(self.n_epochs)):
            loss_sublist = []  # Store individual batch losses for this epoch

            # Training phase
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.model.train()  # Set model to training mode

                z = self.model(x)  # Forward pass
                loss = self.criterion(z, y)  # Compute loss
                loss_sublist.append(loss.item())

                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                self.optimizer.zero_grad()  # Reset gradients

            print(f"\nEpoch {epoch + 1} done")

            # Adjust learning rate if scheduler is defined
            self.scheduler.step()

            # Store average training loss for this epoch
            loss_list.append(np.mean(loss_sublist))

            # Validation phase
            correct = 0
            self.model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                for x_test, y_test in validation_loader:
                    x_test, y_test = x_test.to(self.device), y_test.to(self.device)
                    z = self.model(x_test)
                    _, yhat = torch.max(z.data, 1)
                    correct += (yhat == y_test).sum().item()

            accuracy = correct / n_test
            accuracy_list.append(accuracy)

            # Save best model
            if accuracy > accuracy_best:
                accuracy_best = accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())

            # Print training progress
            if print_:
                print("Learning rate:", self.optimizer.param_groups[0]['lr'])
                print(f"Validation loss (epoch {epoch + 1}): {np.mean(loss_sublist):.4f}")
                print(f"Validation accuracy (epoch {epoch + 1}): {accuracy:.4f}")

        print("\nBest accuracy:", accuracy_best)
        print("Training finished.")

        # Load best model weights before returning
        self.model.load_state_dict(best_model_wts)

        return accuracy_list, loss_list

    def save_model(self, path, name='model.pt'):
        if not name.endswith('.pt'):
            raise ValueError(f"Incorrect extension in '{name}'. Must be '.pt' file.")
        
        torch.save(self.model.state_dict(), os.path.join(path, name))
        print(f"Model saved at {path}")

    def load_model(self, path):
        if not path.endswith('.pt'):
            raise ValueError(f"Incorrect file extension. Must be '.pt' file.")

        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exists.")
        
        # Load trained weights
        self.model.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        print(f"Successfully loaded from {path}")

    def predict(self, image_path, print_=True):
        # Get classes list
        classes = os.listdir("./dataset/train")
        # Set model to evaluation mode
        self.model.eval()
        # Transform image
        input_ = self.transform_image(image_path)
        file_name = image_path.split('/')[-1]
        # Predict
        with torch.no_grad():
            outputs = self.model(input_)
            prediction = torch.argmax(outputs, 1).item()
            predicted_class = classes[prediction]
            
        if print_:
            print(f'\n{file_name} predicted as: {predicted_class}')

        return prediction, predicted_class        

    def transform_image(self, image_path):
        # Read image
        image = Image.open(image_path).convert('RGB')
        # Transform image
        input_tensor = self.val_transform(image).unsqueeze(0)
        return input_tensor

    def print_hypams(self):
        print("Epochs:", self.n_epochs)
        print("Batch size:", self.batch_size)
        print("Learning rate:", self.lr)
        print("Momentum:", self.momentum)
        print("Lr. scheduler:", self.lr_scheduler)
        print("\tBase lr:", self.base_lr)
        print("\tMax lr:", self.max_lr)
        







        