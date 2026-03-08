from PIL import Image


def create_padded_image(image_paths, row_paddings, col_paddings):
    # Load all images into a 2D list
    images = [[Image.open(path) for path in row] for row in image_paths]

    # Calculate dimensions for the grid
    rows = len(images)
    cols = len(images[0])

    # Ensure paddings are sufficient for the grid
    assert len(row_paddings) == cols + 1, "row_paddings length must be columns + 1"
    assert len(col_paddings) == rows + 1, "col_paddings length must be rows + 1"

    # Get individual image dimensions
    img_widths = [max(img.size[0] for img in col) for col in zip(*images)]
    img_heights = [max(img.size[1] for img in row) for row in images]

    # Calculate total dimensions for the combined image
    total_width = sum(img_widths) + sum(row_paddings)
    total_height = sum(img_heights) + sum(col_paddings)

    # Create a blank image with white background
    combined_image = Image.new("RGB", (total_width, total_height), "white")

    # Place each image in the grid with specified paddings
    y_offset = col_paddings[0]
    for row_idx, row in enumerate(images):
        x_offset = row_paddings[0]
        for col_idx, img in enumerate(row):
            # Calculate x and y offsets for current image
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img_widths[col_idx] + row_paddings[col_idx + 1]
        y_offset += img_heights[row_idx] + col_paddings[row_idx + 1]

    return combined_image


# image_paths = [
#     (
#         'icml_imgs/29_10_painted.png',
#         'icml_imgs/29_10_traj_painted.png',
#         'icml_imgs/gciql_seed2/0_traj_painted.png',
#         'icml_imgs/hilp/0_traj_painted.png'
#     ),
#     (
#         'icml_imgs/35_0_painted.png',
#         'icml_imgs/35_0_traj_painted.png',
#         'icml_imgs/gciql_seed2/3_traj_painted.png',
#         'icml_imgs/hilp/3_traj_painted.png'
#     ),
# ]

# image_paths = [
#     (
#     'icml_imgs/appendix/train_maps/cropped_0.png',
#     'icml_imgs/appendix/train_maps/cropped_1.png',
#     'icml_imgs/appendix/train_maps/cropped_2.png',
#     'icml_imgs/appendix/train_maps/cropped_3.png',
#     'icml_imgs/appendix/train_maps/cropped_4.png'
#     ),
#     (
#     'icml_imgs/appendix/train_maps/cropped_0.png',
#     'icml_imgs/appendix/train_maps/cropped_1.png',
#     'icml_imgs/appendix/train_maps/cropped_2.png',
#     'icml_imgs/appendix/train_maps/cropped_3.png',
#     'icml_imgs/appendix/train_maps/cropped_4.png'
#     )
# ]


# row_paddings = [7,7,7]
# col_paddings = [7,7,7,7,7,7]

# output_path = f'icml_imgs/appendix/trajs/train_combined.png'

# output_image = create_padded_image(
#     image_paths=image_paths,
#     row_paddings =col_paddings,
#     col_paddings=row_paddings,
# )

# output_image.save(output_path)


image_paths = [
    (
        "icml_imgs/appendix/trajs/0_pldm_plan.png",
        "icml_imgs/appendix/trajs/0_pldm_traj.png",
        "icml_imgs/appendix/trajs/0_crl_traj.png",
        "icml_imgs/appendix/trajs/0_gcbc_traj.png",
        "icml_imgs/appendix/trajs/0_gciql_traj.png",
        "icml_imgs/appendix/trajs/0_hiql_traj.png",
        "icml_imgs/appendix/trajs/0_hilp_traj.png",
    ),
    (
        "icml_imgs/appendix/trajs/1_pldm_plan.png",
        "icml_imgs/appendix/trajs/1_pldm_traj.png",
        "icml_imgs/appendix/trajs/1_crl_traj.png",
        "icml_imgs/appendix/trajs/1_gcbc_traj.png",
        "icml_imgs/appendix/trajs/1_gciql_traj.png",
        "icml_imgs/appendix/trajs/1_hiql_traj.png",
        "icml_imgs/appendix/trajs/1_hilp_traj.png",
    ),
    (
        "icml_imgs/appendix/trajs/2_pldm_plan.png",
        "icml_imgs/appendix/trajs/2_pldm_traj.png",
        "icml_imgs/appendix/trajs/2_crl_traj.png",
        "icml_imgs/appendix/trajs/2_gcbc_traj.png",
        "icml_imgs/appendix/trajs/2_gciql_traj.png",
        "icml_imgs/appendix/trajs/2_hiql_traj.png",
        "icml_imgs/appendix/trajs/2_hilp_traj.png",
    ),
    (
        "icml_imgs/appendix/trajs/3_pldm_plan.png",
        "icml_imgs/appendix/trajs/3_pldm_traj.png",
        "icml_imgs/appendix/trajs/3_crl_traj.png",
        "icml_imgs/appendix/trajs/3_gcbc_traj.png",
        "icml_imgs/appendix/trajs/3_gciql_traj.png",
        "icml_imgs/appendix/trajs/3_hiql_traj.png",
        "icml_imgs/appendix/trajs/3_hilp_traj.png",
    ),
    (
        "icml_imgs/appendix/trajs/4_pldm_plan.png",
        "icml_imgs/appendix/trajs/4_pldm_traj.png",
        "icml_imgs/appendix/trajs/4_crl_traj.png",
        "icml_imgs/appendix/trajs/4_gcbc_traj.png",
        "icml_imgs/appendix/trajs/4_gciql_traj.png",
        "icml_imgs/appendix/trajs/4_hiql_traj.png",
        "icml_imgs/appendix/trajs/4_hilp_traj.png",
    ),
    (
        "icml_imgs/appendix/trajs/5_pldm_plan.png",
        "icml_imgs/appendix/trajs/5_pldm_traj.png",
        "icml_imgs/appendix/trajs/5_crl_traj.png",
        "icml_imgs/appendix/trajs/5_gcbc_traj.png",
        "icml_imgs/appendix/trajs/5_gciql_traj.png",
        "icml_imgs/appendix/trajs/5_hiql_traj.png",
        "icml_imgs/appendix/trajs/5_hilp_traj.png",
    ),
    # (
    #     'icml_imgs/appendix/trajs/6_pldm_plan.png',
    #     'icml_imgs/appendix/trajs/6_pldm_traj.png',
    #     'icml_imgs/appendix/trajs/6_crl_traj.png',
    #     'icml_imgs/appendix/trajs/6_gcbc_traj.png',
    #     'icml_imgs/appendix/trajs/6_gciql_traj.png',
    #     'icml_imgs/appendix/trajs/6_hiql_traj.png',
    #     'icml_imgs/appendix/trajs/6_hilp_traj.png',
    # ),
    (
        "icml_imgs/appendix/trajs/7_pldm_plan.png",
        "icml_imgs/appendix/trajs/7_pldm_traj.png",
        "icml_imgs/appendix/trajs/7_crl_traj.png",
        "icml_imgs/appendix/trajs/7_gcbc_traj.png",
        "icml_imgs/appendix/trajs/7_gciql_traj.png",
        "icml_imgs/appendix/trajs/7_hiql_traj.png",
        "icml_imgs/appendix/trajs/7_hilp_traj.png",
    ),
    (
        "icml_imgs/appendix/trajs/8_pldm_plan.png",
        "icml_imgs/appendix/trajs/8_pldm_traj.png",
        "icml_imgs/appendix/trajs/8_crl_traj.png",
        "icml_imgs/appendix/trajs/8_gcbc_traj.png",
        "icml_imgs/appendix/trajs/8_gciql_traj.png",
        "icml_imgs/appendix/trajs/8_hiql_traj.png",
        "icml_imgs/appendix/trajs/8_hilp_traj.png",
    ),
]

col_captions = ["PLDM Plan", "PLDM Path", "GCIQL Path", "HILP Path"]
row_captions = ["Layout 1", "Layout 2"]
row_paddings = [0, 7, 7, 7, 7, 7, 7, 7, 0]
col_paddings = [0, 7, 7, 7, 7, 7, 7, 0]

output_path = f"icml_imgs/appendix/trajs/combined.png"

output_image = create_padded_image(
    image_paths=image_paths,
    row_paddings=col_paddings,
    col_paddings=row_paddings,
)


output_image.save(output_path)
