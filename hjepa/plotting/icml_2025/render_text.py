import matplotlib.pyplot as plt

# Create a figure with tight layout
fig, ax = plt.subplots(figsize=(4, 2))  # Adjust size as needed
ax.set_axis_off()  # Hide the axis for clean output

# Render text
ax.text(
    0.5,
    0.5,
    "Train on Diverse Maps           Test on Unseen Maps",
    fontsize=14,
    fontweight="bold",
    color="black",
    ha="center",
    va="center",
)

# Save as PNG without extra white space
plt.savefig(
    "icml_imgs/maze_text_2.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True,
)
