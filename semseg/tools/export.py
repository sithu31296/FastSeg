import argparse
from pathlib import Path

import onnx
import torch
import yaml
from onnxsim import simplify

from semseg.datasets import *  # noqa: F403
from semseg.models import *  # noqa: F403


def export_onnx(model, inputs, file):
    torch.onnx.export(
        model,
        inputs,
        f"{cfg['TEST']['MODEL_PATH'].split('.')[0]}.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
    )
    onnx_model = onnx.load(f"{file}.onnx")
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    onnx.save(onnx_model, f"{file}.onnx")
    assert check, "Simplified ONNX model could not be validated"
    print(f"ONNX model saved to {file}.onnx")


def export_coreml(model, inputs, file):
    try:
        import coremltools as ct

        ts_model = torch.jit.trace(model, inputs, strict=True)
        ct_model = ct.convert(
            ts_model,
            inputs=[
                ct.ImageType(
                    "image", shape=inputs.shape, scale=1 / 255.0, bias=[0, 0, 0]
                )
            ],
        )
        ct_model.save(f"{file}.mlmodel")
        print(f"CoreML model saved to {file}.mlmodel")
    except Exception:
        print(
            "Please install coremltools to export to CoreML.\n`pip install coremltools`"
        )


def main(cfg):
    model = eval(cfg["MODEL"]["NAME"])(
        cfg["MODEL"]["BACKBONE"], len(eval(cfg["DATASET"]["NAME"]).PALETTE)
    )
    model.load_state_dict(torch.load(cfg["TEST"]["MODEL_PATH"], map_location="cpu"))
    model.eval()

    inputs = torch.randn(1, 3, *cfg["TEST"]["IMAGE_SIZE"])
    file = cfg["TEST"]["MODEL_PATH"].split(".")[0]

    export_onnx(model, inputs, file)
    export_coreml(model, inputs, file)
    print("Finished converting.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/custom.yaml")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    save_dir = Path(cfg["SAVE_DIR"])
    save_dir.mkdir(exist_ok=True)

    main(cfg)
