from tabulate import tabulate
from semseg import models
from semseg import datasets
from semseg.models import backbones, heads


def show_models():
    model_names = models.__all__
    numbers = list(range(1, len(model_names) + 1))
    print(tabulate({"No.": numbers, "Model Names": model_names}, headers="keys"))


def show_backbones():
    backbone_names = backbones.__all__
    variants = []
    for name in backbone_names:
        try:
            variants.append(list(eval(f"backbones.{name.lower()}_settings").keys()))
        except Exception:
            variants.append("-")
    print(
        tabulate(
            {"Backbone Names": backbone_names, "Variants": variants}, headers="keys"
        )
    )


def show_heads():
    head_names = heads.__all__
    numbers = list(range(1, len(head_names) + 1))
    print(tabulate({"No.": numbers, "Heads": head_names}, headers="keys"))


def show_datasets():
    dataset_names = datasets.__all__
    numbers = list(range(1, len(dataset_names) + 1))
    print(tabulate({"No.": numbers, "Datasets": dataset_names}, headers="keys"))
