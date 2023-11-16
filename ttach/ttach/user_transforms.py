import kornia.augmentation as kaug
import torch
import torchvision.transforms as T

from .base import DualTransform, ImageOnlyTransform


class Rotation(DualTransform):
    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            assert isinstance(image, tuple)
            res = []
            for img in image:
                res.append(T.functional.rotate(img, 30, fill=0))
            image = tuple(res)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            assert isinstance(mask, tuple)
            res = []
            for img in mask:
                res.append(T.functional.rotate(img, -30, fill=float("nan")))
            mask = tuple(res)
        return mask


class Upscale(DualTransform):
    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            assert isinstance(image, tuple)
            res = []
            for img in image:
                res.append(
                    T.functional.affine(
                        img, angle=0, translate=[0, 0], scale=1.1, shear=0
                    )
                )
            image = tuple(res)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            assert isinstance(mask, tuple)
            res = []
            for img in mask:
                res.append(
                    T.functional.affine(
                        img,
                        angle=0,
                        translate=[0, 0],
                        scale=(1 / 1.1),
                        shear=0,
                        fill=float("nan"),
                    )
                )
            mask = tuple(res)
        return mask


class Downscale(DualTransform):
    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            assert isinstance(image, tuple)
            res = []
            for img in image:
                res.append(
                    T.functional.affine(
                        img, angle=0, translate=[0, 0], scale=0.9, shear=0, fill=0
                    )
                )
            image = tuple(res)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            assert isinstance(mask, tuple)
            res = []
            for img in mask:
                res.append(
                    T.functional.affine(
                        img, angle=0, translate=[0, 0], scale=(1 / 0.9), shear=0
                    )
                )
            mask = tuple(res)
        return mask


class ImageNetNormalize(ImageOnlyTransform):
    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = self._ImageNetNormalize()(image)
        return image

    class _ImageNetNormalize(torch.nn.Module):
        def __init__(self):
            super().__init__()

            image_net_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
            image_net_std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

            self.image_net_mean = torch.nn.parameter.Parameter(
                image_net_mean, requires_grad=False
            )
            self.image_net_std = torch.nn.parameter.Parameter(
                image_net_std, requires_grad=False
            )

        def __call__(self, x):
            # x.shape == B, C, H, W
            self.to(x.device)

            x_mean = x.mean(dim=[-1, -2])[..., None, None]
            x_std = x.std(dim=[-1, -2])[..., None, None]

            std_ratio = self.image_net_std / x_std

            x = ((x - x_mean) * std_ratio) + self.image_net_mean

            x[x > 1] = 1
            x[x < 0] = 0

            return x


class GaussianNoise(ImageOnlyTransform):
    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = kaug.RandomGaussianNoise(mean=0.0, std=0.05, p=1.0, keepdim=True)(
                image
            )
        return image


class Grayscale(ImageOnlyTransform):
    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = kaug.RandomGrayscale(p=1.0, keepdim=True)(image)
        return image
