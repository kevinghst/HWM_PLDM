import torch

from matplotlib import pyplot as plt

from hjepa.logger import Logger


def show_image(img: torch.Tensor):
    img = img.detach().cpu().numpy().sum(axis=0)
    img = (img + 4) / 8
    img = img.clip(0, 1)
    img = 1 - img
    plt.imshow(img, cmap="gray")
    plt.axis("off")


def plot_reconstructions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_images: int = 10,
    suffix: str = "",
    notebook: bool = False,
):
    # select a few
    predictions = predictions[:n_images]
    targets = targets[:n_images]

    T = targets.shape[1]
    fig = plt.figure(dpi=400, figsize=(T, n_images * 2))

    for i in range(n_images):
        for t in range(targets.shape[1]):
            plt_index_pred = (2 * T) * i + t + 1
            plt_index_gt = (2 * T) * i + t + T + 1
            plt.subplot(n_images * 2, T, plt_index_pred)
            show_image(predictions[i, t])
            plt.title("t={} pred".format(t), fontsize=6)
            plt.subplot(n_images * 2, T, plt_index_gt)
            show_image(targets[i, t])
            plt.title("t={} GT".format(t), fontsize=6)

    plt.tight_layout()

    if not notebook:
        Logger.run().log_figure(fig, f"reconstructions{suffix}")
        plt.close(fig)
