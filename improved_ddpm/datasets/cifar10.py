import os
import torchvision
from PIL import Image

def save_cifar10_images(output_dir):
    dataset = torchvision.datasets.CIFAR10(root='/home/zhujunyi/nfs/dataset/cifar10', train=True, download=True)
    
    for i, (image, label) in enumerate(dataset):
        label_dir = os.path.join(output_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        image_path = os.path.join(label_dir, f"{i}.png")
        image.save(image_path)

# Example usage
output_dir = "/home/zhujunyi/nfs/dataset/cifar10/cifar10_png"
save_cifar10_images(output_dir)
