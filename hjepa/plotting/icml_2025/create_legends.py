import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def create_legend(color_dict, filename="legend.png", dpi=300):
    """
    Create a vertical legend figure using a color dictionary and save it as an image.

    If a color value is None, the label is displayed without a color box and with extra spacing.

    Args:
        color_dict (dict): Dictionary where keys are labels and values are hex color codes or None.
        filename (str): Output filename (default: 'legend.png').
        dpi (int): Resolution of the saved image.
    """
    # Create legend handles (with extra spacing for None entries)
    patches = []
    for label, color in color_dict.items():
        if color:  # Only add a colored patch if a valid color is provided
            patches.append(mpatches.Patch(color=color, label=label))
        else:
            patches.append("")  # Insert a blank space for extra spacing
            patches.append(
                plt.Line2D([], [], linestyle="", label=label)
            )  # Add text-only label

    # Create figure
    fig, ax = plt.subplots(
        figsize=(2.5, len(color_dict) * 0.6)
    )  # Adjust height dynamically
    ax.set_axis_off()  # Remove axes

    # Add legend (handling text-only labels and spacing)
    legend = ax.legend(
        handles=[
            (
                p
                if isinstance(p, (mpatches.Patch, plt.Line2D))
                else plt.Line2D([], [], linestyle="", label=p)
            )
            for p in patches
        ],
        loc="center",
        frameon=False,
    )

    # Save as image
    plt.savefig(
        filename, dpi=dpi, bbox_inches="tight", pad_inches=0.2, transparent=True
    )
    plt.show()


def create_horizontal_legend(color_dict, filename="legend_horizontal.png", dpi=300):
    """
    Create a horizontal legend figure using a color dictionary and save it as an image.

    The legend is structured with color boxes in the top row and corresponding labels below them.
    Long labels (like 'PLDM Plan') can be split into multiple lines.

    Args:
        color_dict (dict): Dictionary where keys are labels and values are hex color codes or None.
        filename (str): Output filename (default: 'legend_horizontal.png').
        dpi (int): Resolution of the saved image.
    """
    labels = list(color_dict.keys())
    colors = list(color_dict.values())

    num_items = len(labels)
    box_size = 0.6  # Size of each color box
    text_offset = 0  # Offset between boxes and text
    text_size = 20  # Adjust font size for better readability

    fig, ax = plt.subplots(figsize=(num_items * 1.5, 2))  # Adjust width dynamically
    ax.set_xlim(0, num_items)
    ax.set_ylim(0, 2)
    ax.set_axis_off()  # Remove axes

    # Draw color boxes
    for i, color in enumerate(colors):
        if color:  # Draw color box
            rect = mpatches.Rectangle(
                (i, 1), 1, box_size, color=color, ec="black", lw=0.5
            )
            ax.add_patch(rect)

    # Add text labels below each box, handling multi-line text
    for i, label in enumerate(labels):
        if label == "PLDM Plan":  # Manually break it into two rows
            label = "PLDM\nPlan"
        ax.text(
            i + 0.5,
            0.5 - text_offset,
            label,
            ha="center",
            va="top",
            fontsize=text_size,
            wrap=True,
            linespacing=1.2,
        )

    # Save as image
    plt.savefig(
        filename, dpi=dpi, bbox_inches="tight", pad_inches=0.2, transparent=True
    )
    plt.show()


# Given color mapping dictionary
method_colors = {
    "PLDM Plan": "#FF00FF",
    "PLDM": "#d62728",
    # "GCBC": "#8c564b",
    # "GCIQL": "#ff7f0e",
    "HILP": "#2ca02c",
    "HIQL": "#9467bd",
    "CRL": "#1f77b4",
    "Goal": None,
}

filename = "icml_imgs/main_maze/horizontal_legend.png"

# Generate the legend
create_horizontal_legend(method_colors, filename=filename)
