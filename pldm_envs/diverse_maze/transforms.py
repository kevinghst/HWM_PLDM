from torchvision import transforms
from torchvision.transforms import functional as F


class CustomCrop:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return F.crop(img, self.top, self.left, self.height, self.width)


class RemoveRedBall:
    def __init__(self):
        pass

    def __call__(self, img):
        # NOTE: This is only tested for medium_diverse. May or may not generalize
        img_t = transforms.ToTensor()(img)
        img_t[:, -39] = img_t[:, -39, -1].unsqueeze(-1)
        img_t[:, -38:] = img_t[:, -1].unsqueeze(1)
        img_t[:, -53:-39] = img_t[:, 0].unsqueeze(1)
        img_new = transforms.ToPILImage()(img_t)
        return img_new


umaze_transforms = transforms.Compose(
    [
        transforms.CenterCrop(200),
        transforms.Resize(64),
    ]
)

medium_transforms = transforms.Compose(
    [
        transforms.CenterCrop(300),
        transforms.Resize(64),
    ]
)

large_transforms = transforms.Compose(
    [
        transforms.CenterCrop(370),
        transforms.Resize(64),
    ]
)

# deprecated
# medium_diverse_transforms = transforms.Compose(
#     [
#         RemoveRedBall(),
#         CustomCrop(10, 10, 486, 486),
#         transforms.Resize(81),
#     ]
# )

large_diverse_transforms = transforms.Compose(
    [
        transforms.CenterCrop(426),
        transforms.Resize(98),
    ]
)

medium_diverse_transforms = transforms.Compose(
    [
        transforms.CenterCrop(386),
        transforms.Resize(81),
    ]
)

small_diverse_transforms = transforms.Compose(
    [
        transforms.CenterCrop(346),
        transforms.Resize(64),
    ]
)


def select_transforms(env_name):
    if "umaze" in env_name:
        return umaze_transforms
    elif "small" in env_name:
        return small_diverse_transforms
    elif "large" in env_name:
        return large_diverse_transforms
    elif "medium" in env_name:
        if "diverse" in env_name:
            return medium_diverse_transforms
        else:
            return medium_transforms
    elif "large" in env_name:
        return large_transforms
