import torchvision
import torchvision.transforms.v2 as transforms

torchvision.disable_beta_transforms_warning()


def init_aug():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=90,
                fill=float("nan"),
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            # transforms.ElasticTransform(fill =-1),
        ]
    )
    return transform
