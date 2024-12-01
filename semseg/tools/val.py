import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import torch
import yaml
from tabulate import tabulate
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from semseg.augmentations import get_val_augmentation
from semseg.datasets import *  # noqa: F403
from semseg.metrics import Metrics
from semseg.models import *  # noqa: F403
from semseg.utils.utils import setup_cudnn


@torch.no_grad()
def evaluate(model, dataloader, device):
    print("Evaluating...")
    model.eval()
    metrics = Metrics(
        dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device
    )

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).softmax(dim=1)
        metrics.update(preds, labels)

    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return acc, macc, f1, mf1, ious, miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = (
                int(math.ceil(new_H / 32)) * 32,
                int(math.ceil(new_W / 32)) * 32,
            )
            scaled_images = F.interpolate(
                images, size=(new_H, new_W), mode="bilinear", align_corners=True
            )
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=True
            )
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=True
                )
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg["DEVICE"])

    eval_cfg = cfg["EVAL"]
    transform = get_val_augmentation(eval_cfg["IMAGE_SIZE"])
    dataset = eval(cfg["DATASET"]["NAME"])(cfg["DATASET"]["ROOT"], "val", transform)
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg["MODEL_PATH"])
    if not model_path.exists():
        model_path = (
            Path(cfg["SAVE_DIR"])
            / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
        )
    print(f"Evaluating {model_path}...")

    model = eval(cfg["MODEL"]["NAME"])(cfg["MODEL"]["BACKBONE"], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model = model.to(device)

    if eval_cfg["MSF"]["ENABLE"]:
        acc, macc, f1, mf1, ious, miou = evaluate_msf(
            model,
            dataloader,
            device,
            eval_cfg["MSF"]["SCALES"],
            eval_cfg["MSF"]["FLIP"],
        )
    else:
        acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device)

    table = {
        "Class": list(dataset.CLASSES) + ["Mean"],
        "IoU": ious + [miou],
        "F1": f1 + [mf1],
        "Acc": acc + [macc],
    }

    print(tabulate(table, headers="keys"))


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Evaluate the model",
    )
    parser.add_argument("--cfg", type=str, default="configs/custom.yaml")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
