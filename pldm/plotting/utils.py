import tempfile
import base64

import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import IPython.display as display
from tqdm import tqdm
import math
from PIL import Image, ImageDraw
from pldm import data


def load_uniform_umaze():
    with torch.no_grad():
        uniform_ds = torch.load("/vast/wz1232/d4rl_data_point_umaze/data.p")
        uniform_obs = np.stack([x["observations"] for x in uniform_ds]).reshape(-1, 4)
    return uniform_obs


def load_uniform_large():
    with torch.no_grad():
        uniform_ds = torch.load("/checkpoint/vladsobal/data/d4rl/uniform_point_large.p")
        uniform_obs = np.stack([x["observations"] for x in uniform_ds]).reshape(-1, 4)
    return uniform_obs


def load_expert_umaze():
    env_name = "maze2d-umaze-v1"
    env2 = data.ant_draw.load_environment(env_name)
    ds = data.ant_draw.antmaze_get_dataset(
        env2
    )  # use this to get dataset, which contains some fixes for poorly formatted d4rl data
    return ds["observations"]


def load_expert_medium():
    env_name = "maze2d-medium-v1"
    env2 = data.ant_draw.load_environment(env_name)
    ds = data.ant_draw.antmaze_get_dataset(
        env2
    )  # use this to get dataset, which contains some fixes for poorly formatted d4rl data
    return ds["observations"]


def load_expert_large():
    env_name = "maze2d-large-v1"
    env2 = data.ant_draw.load_environment(env_name)
    ds = data.ant_draw.antmaze_get_dataset(
        env2
    )  # use this to get dataset, which contains some fixes for poorly formatted d4rl data
    return ds["observations"]


def load_expert_ant_umaze():
    env_name = "antmaze-umaze-diverse-v1"
    env2 = data.ant_draw.load_environment(env_name)
    ds = data.ant_draw.antmaze_get_dataset(
        env2
    )  # use this to get dataset, which contains some fixes for poorly formatted d4rl data
    return ds["observations"]


def load_expert_ant_medium():
    env_name = "antmaze-medium-diverse-v1"
    env2 = data.ant_draw.load_environment(env_name)
    ds = data.ant_draw.antmaze_get_dataset(
        env2
    )  # use this to get dataset, which contains some fixes for poorly formatted d4rl data
    return ds["observations"]


def load_expert_ant_large():
    env_name = "antmaze-large-diverse-v1"
    env2 = data.ant_draw.load_environment(env_name)
    ds = data.ant_draw.antmaze_get_dataset(
        env2
    )  # use this to get dataset, which contains some fixes for poorly formatted d4rl data
    return ds["observations"]


def select_subset(x, D=2000):
    indices = np.random.choice(x.shape[0], D, replace=False)
    return x[indices]


def scatter_subset(x, y, c=None, D=20000, *args, **kwargs):
    # fix rng seed
    rng = np.random.default_rng(0)
    # indices = np.random.choice(x.shape[0], D, replace=False)
    indices = rng.choice(x.shape[0], D, replace=False)
    if c is not None and not isinstance(c, str):
        c = c[indices]
    plt.scatter(x[indices], y[indices], c=c, *args, **kwargs)


def show_gif(plot_stuff, N, fps):
    # Generate the frames
    frames = []
    for idx in tqdm(range(N)):
        # Call the plot_stuff function to generate the plot for the current index
        plot_stuff(idx)

        # Create a buffer to save the plot image
        buffer = tempfile.NamedTemporaryFile(suffix=".png")

        # Save the plot image to the buffer
        plt.savefig(buffer.name, format="png")

        # Rewind the buffer
        buffer.seek(0)

        # Open the image using PIL
        image = Image.open(buffer)
        image.load()

        # Append the image to the frames list
        frames.append(image)

        # Close the buffer to clean up the temporary file
        buffer.close()

        plt.close()

    # Create the GIF
    gif_data = b""
    with tempfile.NamedTemporaryFile(suffix=".gif") as gif_file:
        frames[0].save(
            gif_file.name,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=int(1000 / fps),
            loop=0,
        )
        gif_file.seek(0)
        gif_data = gif_file.read()

    gif_data_base64 = base64.b64encode(gif_data).decode("ascii")

    # Display the GIF in the notebook
    display.display(
        display.HTML('<img src="data:image/gif;base64,{0}">'.format(gif_data_base64))
    )


def paint_at_xy(image, xy, dot_radius=3, color="red"):
    if color == "red":
        fill = (255, 0, 0)
    elif color == "orange":
        fill = (204, 132, 0)
    elif color == "blue":
        fill = (0, 0, 200)
    elif color == "black":
        fill = (0, 0, 0)
    elif color == "purple":
        fill = (128, 0, 128)
    else:
        raise ValueError(f"Invalid color: {color}")

    # Convert coordinate to integers (pixel indices)
    x, y = map(int, xy.tolist())

    # Draw the red dot on the image
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill=fill
    )
    return image


def paint_at_xy_star(image, xy, dot_radius=3, color="red"):
    # Define colors
    if color == "red":
        fill = (255, 0, 0)
    elif color == "orange":
        fill = (204, 132, 0)
    elif color == "blue":
        fill = (0, 0, 200)
    elif color == "black":
        fill = (0, 0, 0)
    elif color == "purple":
        fill = (128, 0, 128)
    elif color.startswith("#") and len(color) in {
        7,
        9,
    }:  # Supports #RRGGBB or #RRGGBBAA
        fill = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    else:
        raise ValueError(f"Invalid color: {color}")

    # Convert coordinate to integers (pixel indices)
    x, y = map(int, xy.tolist())

    # Draw the star
    draw = ImageDraw.Draw(image)

    # Calculate the points of the star
    num_points = 5  # Number of points for the star
    outer_radius = dot_radius  # Outer radius of the star
    inner_radius = dot_radius // 2  # Inner radius of the star
    points = []

    for i in range(num_points * 2):  # Double the points (outer + inner)
        angle = i * math.pi / num_points  # Angle of the point
        r = (
            outer_radius if i % 2 == 0 else inner_radius
        )  # Alternate between outer and inner radius
        px = x + r * math.cos(angle)
        py = y + r * math.sin(angle)
        points.append((px, py))

    # Draw the star
    draw.polygon(points, fill=fill)

    return image


def paint_series(
    image,
    series,
    dot_radius=3,
    dot_opacity=128,
    line_opacity=255,
    color="red",
    mark_dot=False,
):
    """
    Paint a series of dots and lines on an image with transparency and controlled color.

    Args:
        image (PIL.Image): The original image.
        series (torch.Tensor): A tensor of shape (n, 2) representing coordinates.
        dot_radius (int): The radius of the dots.
        dot_opacity (int): The opacity of the dots (0-255).
        line_opacity (int): The opacity of the lines (0-255).
        color (str): Either 'red', 'blue', 'green', or a hex color code (e.g., '#FF5733').

    Returns:
        PIL.Image: The image with the series painted.
    """

    # Create an overlay image for transparency
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Convert color name or hex to RGBA with specified opacity
    predefined_colors = {
        "red": (180, 0, 0),
        "blue": (0, 0, 180),
        "green": (0, 180, 0),
        "orange": (204, 132, 0),
    }

    if color in predefined_colors:
        base_color = predefined_colors[color]
    elif color.startswith("#") and len(color) in {
        7,
        9,
    }:  # Supports #RRGGBB or #RRGGBBAA
        base_color = tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    else:
        raise ValueError(f"Unsupported color format: {color}")

    dot_color = (*base_color, dot_opacity)
    line_color = (*base_color, line_opacity)

    for i in range(len(series)):
        # Get current coordinate
        x, y = map(int, series[i].tolist())

        if mark_dot:
            # Draw the semi-transparent dot
            overlay_draw.ellipse(
                (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
                fill=dot_color,
            )

        # Connect with a line to the next point (if not the last point)
        if i < len(series) - 1:
            x_next, y_next = map(int, series[i + 1].tolist())
            overlay_draw.line((x, y, x_next, y_next), fill=line_color, width=3)

    # Blend the overlay with the original image
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    return image.convert("RGB")  # Convert back to RGB if needed
