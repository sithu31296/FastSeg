import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from semseg.models import *  # noqa: F403


def main(
    model_name: str,
    backbone_name: str,
    image_size: list,
    num_classes: int,
    device: str,
):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )
    inputs = torch.randn(1, 3, *image_size).to(device)
    model = eval(model_name)(backbone_name, num_classes)
    model = model.to(device)
    model.eval()

    print(flop_count_table(FlopCountAnalysis(model, inputs)))

    total_time = 0.0
    for _ in range(10):
        tic = time.perf_counter()
        model(inputs)
        toc = time.perf_counter()
        total_time += toc - tic
    total_time /= 10
    print(f"Inference time: {total_time*1000:.2f}ms")
    print(f"FPS: {1/total_time}")


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Benchmarking the model",
    )
    parser.add_argument("--model-name", type=str, default="SegFormer")
    parser.add_argument("--backbone-name", type=str, default="MiT-B0")
    parser.add_argument("--image-size", type=list, default=[512, 512])
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(
        args.model_name,
        args.backbone_name,
        args.image_size,
        args.num_classes,
        args.device,
    )
