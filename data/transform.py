from torchvision import transforms
import torch


class ToPILImageIfTensor:
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            # Check if the input is a batch of images
            if pic.ndimension() == 4:
                # Convert each image in the batch to PIL
                return [transforms.ToPILImage()(img) for img in pic]
            elif pic.ndimension() == 3:
                return transforms.ToPILImage()(pic)
            else:
                raise ValueError("pic should be 3 or 4 dimensional. Got {} dimensions.".format(pic.ndimension()))
        return pic


class ApplyTransformToBatch:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch):
        if isinstance(batch, list):
            return torch.stack([self.transform(img) for img in batch])
        else:
            return self.transform(batch)


# Define your transformations
base_transforms = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])



class Test2TrainTransform:

    def __init__(self):
        self.transform = transforms.Compose([
            ToPILImageIfTensor(),
            ApplyTransformToBatch(base_transforms)
        ])

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


class TrainTransform:

    def __init__(self):
        self.transform = base_transforms

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


class TestTransform:

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])

    def __call__(self, sample):
        return self.transform(sample)
