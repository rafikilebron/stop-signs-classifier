# Deep learning
from torchvision.datasets import ImageFolder
from torchvision import transforms

# File management
import os

# Data processing
import shutil
import random

class DatasetCreator:
    def __init__(self, output_dir, train_ratio=0.9, n_classes=2, gray=False):
        self.source_dir     = ""
        self.output_dir     = output_dir
        self.train_ratio    = train_ratio
        self.n_classes      = n_classes
        self.gray           = gray

    def create_dataset(self, images_path, train_transform, val_transform):
        print("-"*40)
        print("DATASET CREATOR\nCreating dataset...")
        # Create label -> images map
        self.list_label_to_images(images_path)
        # Split images
        self.split_dataset()
        # Apply transformations on dataset
        train_dataset, val_dataset = self.transform_dataset(train_transform, val_transform)
        print("Successfully created dataset")
        print("-"*40)
        return train_dataset, val_dataset

    def list_label_to_images(self, images_path):
        self.source_dir = images_path
        self.label_to_images = {}
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        # Verify if source dir exists
        if not os.path.exists(self.source_dir):
            raise FileNotFoundError(f"Directory '{self.source_dir}' does not exist.")

        # Verify if source directory is not a file
        if not os.path.isdir(self.source_dir):
            raise NotADirectoryError(f"Directory '{self.source_dir}' is not a directory.")

        # Read directory
        try:
            # Loop over every class
            for label in os.listdir(self.source_dir):
                label_path = os.path.join(self.source_dir, label)

                # Ignore if not a folder
                if not os.path.isdir(label_path):
                    continue

                # Loop over every image
                for file_path in os.listdir(os.path.join(self.source_dir, label)):
                    # Verify if file is an image
                    if not file_path.endswith(valid_ext):
                        continue

                    # Store file name
                    self.label_to_images.setdefault(str(label), []).append(file_path)

        except PermissionError as e:
            raise PermissionError(f"Cannot access '{self.source_dir}'") from e

        if len(self.label_to_images) == 0:
            raise ValueError("No images found in '{self.source_dir}'")

    def split_dataset(self):
        n_train = 0
        n_validation = 0

        print("Splitting dataset...")

        # Clean previous dataset
        shutil.rmtree(self.output_dir, ignore_errors=True)

        # Divide train/validation
        for label, image_list in self.label_to_images.items():
            # Shuffle the list of images
            random.shuffle(image_list)
            # Calculate the number of training images
            train_cutoff = int(len(image_list) * self.train_ratio)
            val_cutoff = int(len(image_list) - train_cutoff)
            n_train += train_cutoff
            n_validation += val_cutoff
            # Split images
            train_images = image_list[:train_cutoff]
            val_images = image_list[train_cutoff:]

            for split, split_images in zip(['train', 'val'], [train_images, val_images]):
                out_path = os.path.join(self.output_dir, split, label)
                os.makedirs(out_path, exist_ok=True)
                # Copy each image
                for img_name in split_images:
                    src = os.path.join(self.source_dir, label, img_name)
                    dst = os.path.join(self.output_dir, split, label)
                    shutil.copy2(src, dst)

        print(f"Train/Val split complete -> {self.train_ratio*100:.2f}% training")
        print(f"{n_train} train samples\n{n_validation} validation samples")

    def transform_dataset(self, train_transform, val_transform):
        print("Transforming dataset...")

        # Include grayscale transformation
        if self.gray:
            train_transform.transforms.insert(1, transforms.Grayscale(num_output_channels=1))
            val_transform.transforms.insert(1, transforms.Grayscale(num_output_channels=1))

        train_dataset = ImageFolder(self.output_dir+'/train', transform=train_transform)
        val_dataset = ImageFolder(self.output_dir+'/val', transform=val_transform)

        return train_dataset, val_dataset