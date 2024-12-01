import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid

from semseg.augmentations import Compose, Normalize, RandomResizedCrop


def visualize_dataset_sample(dataset, root, split="val", batch_size=4):
    transform = Compose([RandomResizedCrop((512, 512), scale=(1.0, 1.0)), Normalize()])

    dataset = dataset(root, split=split, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    image, label = next(iter(dataloader))

    print(f"Image Shape\t: {image.shape}")
    print(f"Label Shape\t: {label.shape}")
    print(f"Classes\t\t: {label.unique().tolist()}")

    label[label == -1] = 0
    label[label == 255] = 0
    labels = [dataset.PALETTE[lbl.to(int)].permute(2, 0, 1) for lbl in label]
    labels = torch.stack(labels)

    inv_normalize = T.Normalize(
        mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
        std=(1 / 0.229, 1 / 0.224, 1 / 0.225),
    )
    image = inv_normalize(image)
    image *= 255
    images = torch.vstack([image, labels])

    plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
    plt.show()


colors = [
    [120, 120, 120],
    [180, 120, 120],
    [6, 230, 230],
    [80, 50, 50],
    [4, 200, 3],
    [120, 120, 80],
    [140, 140, 140],
    [204, 5, 255],
    [230, 230, 230],
    [4, 250, 7],
    [224, 5, 255],
    [235, 255, 7],
    [150, 5, 61],
    [120, 120, 70],
    [8, 255, 51],
    [255, 6, 82],
    [143, 255, 140],
    [204, 255, 4],
    [255, 51, 7],
    [204, 70, 3],
    [0, 102, 200],
    [61, 230, 250],
    [255, 6, 51],
    [11, 102, 255],
    [255, 7, 71],
    [255, 9, 224],
    [9, 7, 230],
    [220, 220, 220],
    [255, 9, 92],
    [112, 9, 255],
    [8, 255, 214],
    [7, 255, 224],
    [255, 184, 6],
    [10, 255, 71],
    [255, 41, 10],
    [7, 255, 255],
    [224, 255, 8],
    [102, 8, 255],
    [255, 61, 6],
    [255, 194, 7],
    [255, 122, 8],
    [0, 255, 20],
    [255, 8, 41],
    [255, 5, 153],
    [6, 51, 255],
    [235, 12, 255],
    [160, 150, 20],
    [0, 163, 255],
    [140, 140, 140],
    [250, 10, 15],
    [20, 255, 0],
    [31, 255, 0],
    [255, 31, 0],
    [255, 224, 0],
    [153, 255, 0],
    [0, 0, 255],
    [255, 71, 0],
    [0, 235, 255],
    [0, 173, 255],
    [31, 0, 255],
    [11, 200, 200],
    [255, 82, 0],
    [0, 255, 245],
    [0, 61, 255],
    [0, 255, 112],
    [0, 255, 133],
    [255, 0, 0],
    [255, 163, 0],
    [255, 102, 0],
    [194, 255, 0],
    [0, 143, 255],
    [51, 255, 0],
    [0, 82, 255],
    [0, 255, 41],
    [0, 255, 173],
    [10, 0, 255],
    [173, 255, 0],
    [0, 255, 153],
    [255, 92, 0],
    [255, 0, 255],
    [255, 0, 245],
    [255, 0, 102],
    [255, 173, 0],
    [255, 0, 20],
    [255, 184, 184],
    [0, 31, 255],
    [0, 255, 61],
    [0, 71, 255],
    [255, 0, 204],
    [0, 255, 194],
    [0, 255, 82],
    [0, 10, 255],
    [0, 112, 255],
    [51, 0, 255],
    [0, 194, 255],
    [0, 122, 255],
    [0, 255, 163],
    [255, 153, 0],
    [0, 255, 10],
    [255, 112, 0],
    [143, 255, 0],
    [82, 0, 255],
    [163, 255, 0],
    [255, 235, 0],
    [8, 184, 170],
    [133, 0, 255],
    [0, 255, 92],
    [184, 0, 255],
    [255, 0, 31],
    [0, 184, 255],
    [0, 214, 255],
    [255, 0, 112],
    [92, 255, 0],
    [0, 224, 255],
    [112, 224, 255],
    [70, 184, 160],
    [163, 0, 255],
    [153, 0, 255],
    [71, 255, 0],
    [255, 0, 163],
    [255, 204, 0],
    [255, 0, 143],
    [0, 255, 235],
    [133, 255, 0],
    [255, 0, 235],
    [245, 0, 255],
    [255, 0, 122],
    [255, 245, 0],
    [10, 190, 212],
    [214, 255, 0],
    [0, 204, 255],
    [20, 0, 255],
    [255, 255, 0],
    [0, 153, 255],
    [0, 41, 255],
    [0, 255, 204],
    [41, 0, 255],
    [41, 255, 0],
    [173, 0, 255],
    [0, 245, 255],
    [71, 0, 255],
    [122, 0, 255],
    [0, 255, 184],
    [0, 92, 255],
    [184, 255, 0],
    [0, 133, 255],
    [255, 214, 0],
    [25, 194, 194],
    [102, 255, 0],
    [92, 0, 255],
]


def generate_palette(num_classes, background: bool = False):
    random.shuffle(colors)
    if background:
        palette = [[0, 0, 0]]
        palette += colors[: num_classes - 1]
    else:
        palette = colors[:num_classes]
    return np.array(palette)


def draw_text(
    image: torch.Tensor, seg_map: torch.Tensor, labels: list, fontsize: int = 15
):
    image = image.to(torch.uint8)
    font = (
        ImageFont.truetype("assests/Helvetica.ttf", fontsize)
        if os.path.isfile("assests/Helvetica.ttf")
        else ImageFont.load_default(fontsize)
    )
    pil_image = Image.fromarray(image.numpy())
    draw = ImageDraw.Draw(pil_image)

    indices = seg_map.unique().tolist()
    classes = [labels[index] for index in indices]

    for idx, cls in zip(indices, classes):
        mask = seg_map == idx
        mask = mask.squeeze().numpy()
        center = np.median((mask == 1).nonzero(), axis=1)[::-1]
        bbox = draw.textbbox(center, cls, font=font)
        bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
        draw.rectangle(bbox, fill=(255, 255, 255), width=1)
        draw.text(center, cls, fill=(0, 0, 0), font=font)
    return pil_image
