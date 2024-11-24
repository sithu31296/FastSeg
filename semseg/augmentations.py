import torchvision.transforms.functional as TF
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union


class Compose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if mask.ndim == 2:
            assert img.shape[1:] == mask.shape
        else:
            assert img.shape[1:] == mask.shape[1:]

        for transform in self.transforms:
            img, mask = transform(img, mask)

        return img, mask


class Normalize:
    def __init__(
        self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)
    ):
        self.mean = mean
        self.std = std

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        img = img.float()
        img /= 255
        img = TF.normalize(img, self.mean, self.std)
        return img, mask


class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if self.brightness > 0:
            img = TF.adjust_brightness(img, self.brightness)
        if self.contrast > 0:
            img = TF.adjust_contrast(img, self.contrast)
        if self.saturation > 0:
            img = TF.adjust_saturation(img, self.saturation)
        if self.hue > 0:
            img = TF.adjust_hue(img, self.hue)
        return img, mask


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.adjust_sharpness(img, self.sharpness)
        return img, mask


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.autocontrast(img)
        return img, mask


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.gaussian_blur(img, self.kernel_size)
        return img, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits=2):
        self.bits = bits  # 0-8

    def __call__(self, image, label):
        return TF.posterize(image, self.bits), label


class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.seg_fill = seg_fill

    def __call__(self, img, label):
        return TF.affine(
            img,
            self.angle,
            self.translate,
            self.scale,
            self.shear,
            TF.InterpolationMode.BILINEAR,
            0,
        ), TF.affine(
            label,
            self.angle,
            self.translate,
            self.scale,
            self.shear,
            TF.InterpolationMode.NEAREST,
            self.seg_fill,
        )


class RandomRotation:
    def __init__(
        self,
        degrees: float = 10.0,
        p: float = 0.2,
        seg_fill: int = 0,
        expand: bool = False,
    ) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag.
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image.
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            img = TF.rotate(
                img, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0
            )
            mask = TF.rotate(
                mask,
                random_angle,
                TF.InterpolationMode.NEAREST,
                self.expand,
                fill=self.seg_fill,
            )
        return img, mask


class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)


class RandomCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
        """Randomly Crops the image.

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        if random.random() < self.p:
            margin_h = max(H - tH, 0)
            margin_w = max(W - tW, 0)
            y1 = random.randint(0, margin_h + 1)
            x1 = random.randint(0, margin_w + 1)
            y2 = y1 + tH
            x2 = x1 + tW
            img = img[:, y1:y2, x1:x2]
            mask = mask[:, y1:y2, x1:x2]
        return img, mask


class Pad:
    def __init__(
        self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0
    ) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self.size[1] - img.shape[2], self.size[0] - img.shape[1])
        return TF.pad(img, padding), TF.pad(mask, padding, self.seg_fill)


class ResizePad:
    def __init__(
        self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0
    ) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size.
                If size is a sequence, the output size will be matched to this.
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image
        scale_factor = min(tH / H, tW / W) if W > H else max(tH / H, tW / W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask


class Resize:
    def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size.
                If size is a sequence, the output size will be matched to this.
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]

        # scale the image
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H * scale_factor), round(W * scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # make the image divisible by stride
        alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
        return img, mask


class RandomResizedCrop:
    def __init__(
        self,
        size: Union[int, Tuple[int], List[int]],
        scale: Tuple[float, float] = (0.5, 2.0),
        seg_fill: int = 0,
    ) -> None:
        """Resize the input image to the given size."""
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH * ratio), int(tW * 4 * ratio)

        # scale the image
        scale_factor = min(max(scale) / max(H, W), min(scale) / min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # random crop
        margin_h = max(img.shape[1] - tH, 0)
        margin_w = max(img.shape[2] - tW, 0)
        y1 = random.randint(0, margin_h + 1)
        x1 = random.randint(0, margin_w + 1)
        y2 = y1 + tH
        x2 = x1 + tW
        img = img[:, y1:y2, x1:x2]
        mask = mask[:, y1:y2, x1:x2]

        # pad the image
        if img.shape[1:] != self.size:
            padding = [0, 0, tW - img.shape[2], tH - img.shape[1]]
            img = TF.pad(img, padding, fill=0)
            mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask


def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose(
        [
            # ColorJitter(brightness=0.0, contrast=0.5, saturation=0.5, hue=0.5),
            # RandomAdjustSharpness(sharpness_factor=0.1, p=0.5),
            # RandomAutoContrast(p=0.2),
            RandomHorizontalFlip(p=0.5),
            # RandomVerticalFlip(p=0.5),
            # RandomGaussianBlur((3, 3), p=0.5),
            # RandomGrayscale(p=0.5),
            # RandomRotation(degrees=10, p=0.3, seg_fill=seg_fill),
            RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def get_val_augmentation(size: Union[int, Tuple[int], List[int]]):
    return Compose(
        [Resize(size), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )


if __name__ == "__main__":
    h = 230
    w = 420
    img = torch.randn(3, h, w)
    mask = torch.randn(1, h, w)
    aug = Compose(
        [
            RandomResizedCrop((512, 512)),
            # RandomCrop((512, 512), p=1.0),
            # Pad((512, 512))
        ]
    )
    img, mask = aug(img, mask)
    print(img.shape, mask.shape)
