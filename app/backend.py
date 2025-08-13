import glob
import io
import math
import os
import re
import tempfile

import cv2
import matplotlib
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import uniform_filter

matplotlib.use("Agg")
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Reporter:
    def __init__(self):
        self.progress_callback = None
        self.last_was_progress = False

    def set_callback(self, cb):
        if cb is not None:
            self.progress_callback = cb

    def report(self, message: str):
        if self.progress_callback:
            self.progress_callback(message)
            if message.startswith("PROGRESS:"):
                self.last_was_progress = True
            else:
                self.last_was_progress = False
            return

        if message.startswith("PROGRESS:"):
            clean = message.split(":", 1)[1]
            print(clean, end="\r", flush=True)
            self.last_was_progress = True
        else:
            if self.last_was_progress:
                print()
            print(message)
            self.last_was_progress = False


reporter = Reporter()

COLOR_MAP = {
    (65, 155, 223): 1,  # Water
    (57, 125, 73): 2,  # Trees
    (122, 135, 198): 4,  # Flooded Vegetation
    (228, 150, 53): 5,  # Crops
    (196, 40, 27): 7,  # Built Area
    (165, 155, 143): 8,  # Bare Ground
    (227, 226, 195): 11,  # Rangeland
}

CLASS_LABELS = [1, 2, 4, 5, 7, 8, 11]
INDEX_TO_COLOR = {v: k for k, v in COLOR_MAP.items()}
CLASS_ID_TO_LABEL = {
    1: "Water",
    2: "Trees",
    4: "Flooded Vegetation",
    5: "Crops",
    7: "Built Area",
    8: "Bare Ground",
    11: "Rangeland",
}


def refined_lee(img, size=7):
    img = img.astype("float32")
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img * img, size=size)
    var = mean_sq - mean * mean
    sigma_sq = np.mean(var)
    wl = var / (var + sigma_sq)
    return mean + wl * (img - mean)


def _get_utm_crs_from_bounds(bounds, src_crs):
    if not src_crs.is_geographic:
        return src_crs
    lon = (bounds.left + bounds.right) / 2.0
    lat = (bounds.top + bounds.bottom) / 2.0
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return CRS.from_epsg(epsg)


def convert_tiles_inplace(tile_dir: str, is_mask: bool = False, progress_callback=None):
    reporter.set_callback(progress_callback)

    tif_paths = glob.glob(os.path.join(tile_dir, "*.tif"))
    reporter.report(f"Converting tiles in {os.path.basename(tile_dir)}...")

    total_files = len(tif_paths)
    if total_files == 0:
        reporter.report(
            f"No .tif files found to convert in {os.path.basename(tile_dir)}."
        )
        return

    for i, tif_path in enumerate(tif_paths):
        reporter.report(f"PROGRESS:Converting {i + 1}/{total_files} tiles...")

        with rasterio.open(tif_path) as src:
            arr = src.read()
            h, w = src.height, src.width

            if is_mask:
                m = arr[0] > 0
                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgb[m] = 255
            else:
                if arr.shape[0] < 3:
                    raise ValueError(f"{tif_path} has <3 bands")
                rgb = np.stack([arr[i] for i in range(3)], axis=-1)
                if rgb.dtype != np.uint8:
                    mn, mx = float(rgb.min()), float(rgb.max())
                    if mx > mn:
                        rgb = ((rgb.astype(np.float32) - mn) / (mx - mn) * 255).astype(
                            np.uint8
                        )
                    else:
                        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        png_path = tif_path[:-4] + ".png"
        cv2.imwrite(png_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        os.remove(tif_path)

    reporter.report(
        f"Finished converting files in {os.path.basename(tile_dir)} to .png"
    )


def tile_sar_and_optical(
    sar_tif_path: str,
    optical_tif_path: str,
    tile_width: int,
    tile_height: int,
    pixel_size_meters: float,
    base_temp_dir: str,
    progress_callback=None,
) -> tuple[str, str, str]:
    reporter.set_callback(progress_callback)

    reporter.report(
        f"Input images: SAR='{os.path.basename(sar_tif_path)}', optical='{os.path.basename(optical_tif_path)}'"
    )

    with rasterio.open(sar_tif_path) as sar_src:
        orig_crs = sar_src.crs
        orig_transform = sar_src.transform
        orig_bounds = sar_src.bounds
        sar_count = sar_src.count
        sar_dtype = sar_src.dtypes[0]
        sar_nodata = sar_src.nodata

        target_crs = _get_utm_crs_from_bounds(orig_bounds, orig_crs)
        reporter.report(
            f"\nReprojecting SAR from {orig_crs.to_string()} to {target_crs.to_string()} for meter-based tiling..."
        )

        dst_transform, width, height = calculate_default_transform(
            orig_crs,
            target_crs,
            sar_src.width,
            sar_src.height,
            *sar_src.bounds,
            resolution=pixel_size_meters,
        )

        sar_reproj = np.zeros((sar_count, height, width), dtype="float32")
        reproject(
            source=rasterio.band(sar_src, list(range(1, sar_count + 1))),
            destination=sar_reproj,
            src_transform=orig_transform,
            src_crs=orig_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )
        reporter.report(
            f"Reprojected SAR dimensions (pixels): width={width}, height={height}"
        )

        reporter.report("\nApplying Refined Lee filter to reprojected SAR...")
        for b in range(sar_reproj.shape[0]):
            reporter.report(f"PROGRESS:Filtering band {b + 1}/{sar_reproj.shape[0]}...")
            sar_reproj[b] = refined_lee(sar_reproj[b], size=7)
        reporter.report("Refined Lee filter applied.")

        left, top = dst_transform.c, dst_transform.f
        right, bottom = left + dst_transform.a * width, top + dst_transform.e * height
        sar_proj_bounds = rasterio.coords.BoundingBox(
            left=left, bottom=bottom, right=right, top=top
        )
        sar_grid_crs = target_crs
        sar_grid_transform = dst_transform

    reporter.report("\nCreating binary mask from SAR data contours...")
    binary_image = (sar_reproj[0] > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    reporter.report(f"Found {len(contours)} contours in SAR data.")
    binary_mask_reproj = np.zeros_like(binary_image, dtype="uint8")

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(
            binary_mask_reproj, [largest_contour], -1, (255), thickness=cv2.FILLED
        )

        reporter.report("Successfully created binary mask from the largest contour.")
    else:
        reporter.report("Warning: No contours found in SAR data to create a mask.")

    tiles_across = math.ceil(
        (sar_proj_bounds.right - sar_proj_bounds.left)
        / (tile_width * pixel_size_meters)
    )
    tiles_down = math.ceil(
        (sar_proj_bounds.top - sar_proj_bounds.bottom)
        / (tile_height * pixel_size_meters)
    )
    total_tiles = tiles_across * tiles_down
    reporter.report(
        f"\nComputed grid: {tiles_across} tiles across × {tiles_down} tiles down ({total_tiles} total)."
    )

    sar_out_dir = os.path.join(base_temp_dir, "sar_tiles")
    opt_out_dir = os.path.join(base_temp_dir, "optical_tiles")
    mask_out_dir = os.path.join(base_temp_dir, "mask_tiles")
    os.makedirs(sar_out_dir, exist_ok=True)
    os.makedirs(opt_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    with rasterio.open(optical_tif_path) as opt_src:
        reporter.report(
            f"Optical CRS: {opt_src.crs.to_string()}, will reproject per tile into {sar_grid_crs.to_string()}. Beginning tiling..."
        )
        for row in range(tiles_down):
            for col in range(tiles_across):
                reporter.report(
                    f"PROGRESS:Tiling tile {row * tiles_across + col + 1}/{total_tiles}..."
                )

                tlx = sar_proj_bounds.left + col * (tile_width * pixel_size_meters)
                tly = sar_proj_bounds.top - row * (tile_height * pixel_size_meters)
                dst_transform = from_origin(
                    tlx, tly, pixel_size_meters, pixel_size_meters
                )
                tile_filename = f"tile_{row}_{col}.tif"

                sar_dest = np.zeros(
                    (sar_count, tile_height, tile_width), dtype=sar_dtype
                )
                reproject(
                    source=sar_reproj,
                    destination=sar_dest,
                    src_transform=sar_grid_transform,
                    src_crs=sar_grid_crs,
                    dst_transform=dst_transform,
                    dst_crs=sar_grid_crs,
                    resampling=Resampling.nearest,
                    dst_nodata=sar_nodata,
                )
                with rasterio.open(
                    os.path.join(sar_out_dir, tile_filename),
                    "w",
                    driver="GTiff",
                    height=tile_height,
                    width=tile_width,
                    count=sar_count,
                    dtype=sar_dtype,
                    crs=sar_grid_crs,
                    transform=dst_transform,
                    nodata=sar_nodata,
                ) as dst:
                    dst.write(sar_dest)

                opt_dest = np.zeros(
                    (opt_src.count, tile_height, tile_width), dtype=opt_src.dtypes[0]
                )
                reproject(
                    source=rasterio.band(opt_src, list(range(1, opt_src.count + 1))),
                    destination=opt_dest,
                    src_transform=opt_src.transform,
                    src_crs=opt_src.crs,
                    dst_transform=dst_transform,
                    dst_crs=sar_grid_crs,
                    resampling=Resampling.nearest,
                )
                with rasterio.open(
                    os.path.join(opt_out_dir, tile_filename),
                    "w",
                    driver="GTiff",
                    height=tile_height,
                    width=tile_width,
                    count=opt_src.count,
                    dtype=opt_src.dtypes[0],
                    crs=sar_grid_crs,
                    transform=dst_transform,
                    nodata=opt_src.nodata,
                ) as dst:
                    dst.write(opt_dest)

                mask_dest = np.zeros((1, tile_height, tile_width), dtype="uint8")
                reproject(
                    source=binary_mask_reproj,
                    destination=mask_dest,
                    src_transform=sar_grid_transform,
                    src_crs=sar_grid_crs,
                    dst_transform=dst_transform,
                    dst_crs=sar_grid_crs,
                    resampling=Resampling.nearest,
                )
                with rasterio.open(
                    os.path.join(mask_out_dir, tile_filename),
                    "w",
                    driver="GTiff",
                    height=tile_height,
                    width=tile_width,
                    count=1,
                    dtype="uint8",
                    crs=sar_grid_crs,
                    transform=dst_transform,
                    nodata=0,
                ) as dst:
                    dst.write(mask_dest)
    reporter.report("Tiling process complete.")

    reporter.report("\nConverting all tiles to .png for model input...")
    convert_tiles_inplace(
        sar_out_dir, is_mask=False, progress_callback=progress_callback
    )
    convert_tiles_inplace(
        opt_out_dir, is_mask=False, progress_callback=progress_callback
    )
    convert_tiles_inplace(
        mask_out_dir, is_mask=True, progress_callback=progress_callback
    )

    reporter.report("Conversion complete.")
    return sar_out_dir, opt_out_dir, mask_out_dir


class InferenceTransform:
    def __call__(self, image):
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return image


def mask_to_rgb(mask_tensor):
    mask_np = mask_tensor.cpu().numpy()
    rgb_image = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    for class_idx, original_label in enumerate(CLASS_LABELS):
        if original_label in INDEX_TO_COLOR:
            rgb_color = INDEX_TO_COLOR[original_label]
            rgb_image[mask_np == class_idx] = rgb_color
    return Image.fromarray(rgb_image)


def classify_landcover(
    input_dir: str, weights_path: str, base_temp_dir: str, progress_callback=None
) -> str:
    reporter.set_callback(progress_callback)

    reporter.report("\nStarting land cover classification...")
    reporter.report(f"Using device: {DEVICE}")

    num_classes = len(CLASS_LABELS)
    model = smp.Segformer(
        encoder_name="efficientnet-b7", in_channels=3, classes=num_classes
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(DEVICE)))
    model.eval()

    output_dir = os.path.join(base_temp_dir, "land_cover_predictions")
    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    transform = InferenceTransform()

    with torch.no_grad():
        for i, img_path in enumerate(image_files):
            reporter.report(f"PROGRESS:Processing tile {i + 1}/{len(image_files)}...")
            image = Image.open(img_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)
            logits = model(input_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0)
            pred_rgb_image = mask_to_rgb(pred_mask)
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            pred_rgb_image.save(output_path)

    reporter.report("Classification finished. Predictions saved to temporary location.")
    return output_dir


def detect_change_dummy(
    pre_event_dir: str, post_event_dir: str, base_temp_dir: str, progress_callback=None
) -> str:
    reporter.set_callback(progress_callback)
    reporter.report("\nStarting Real Change Detection...")

    # --- Path Definitions ---
    backend_dir = os.path.dirname(__file__)
    cd_dependencies_dir = os.path.join(backend_dir, "cd_dependencies")
    gan_weights_file = os.path.join(cd_dependencies_dir, "mod", "ganmodel.pth")
    # Using 'best_ckpt.pt' as confirmed during our debugging.
    cd_weights_file = os.path.join(cd_dependencies_dir, "mod", "changemodel.pt")
    cd_script_output_dir = os.path.join(base_temp_dir, "cd_script_output")

    for path in [cd_dependencies_dir, gan_weights_file, cd_weights_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file/dir for change detection not found: {path}"
            )

    # Temporarily add the cd_dependencies directory to Python's path for the import
    sys.path.insert(0, cd_dependencies_dir)

    try:
        from fullPredict import main as full_predict_main

        reporter.report("Successfully imported 'fullPredict' module.")

        args_list = [
            "--gan_checkpoint",
            os.path.abspath(gan_weights_file),
            "--cd_checkpoint",
            os.path.abspath(cd_weights_file),
            "--input_sar_dir",
            os.path.abspath(post_event_dir),
            "--input_opt_dir",
            os.path.abspath(pre_event_dir),
            "--output_dir",
            os.path.abspath(cd_script_output_dir),
        ]

        reporter.report("Starting execution of the change detection pipeline...")

        # --- MODIFICATION: Pass the reporter object directly into the function ---
        full_predict_main(args_list, reporter=reporter)

    finally:
        # Always clean up the system path
        if cd_dependencies_dir in sys.path:
            sys.path.remove(cd_dependencies_dir)

    reporter.report("Change Detection process finished.")

    final_predictions_dir = os.path.join(
        os.path.abspath(cd_script_output_dir), "change_detection_results"
    )
    if not os.path.isdir(final_predictions_dir):
        raise NotADirectoryError(
            f"The expected output directory was not created: {final_predictions_dir}"
        )

    return final_predictions_dir


def combine_and_stitch_results(
    lc_pred_dir: str,
    cd_pred_dir: str,
    mask_dir: str,
    base_temp_dir: str,
    progress_callback=None,
) -> str:
    reporter.set_callback(progress_callback)

    reporter.report("\nStarting final combination and stitching...")
    combined_dir = os.path.join(base_temp_dir, "combined_tiles")
    os.makedirs(combined_dir, exist_ok=True)
    lc_tiles = sorted(glob.glob(os.path.join(lc_pred_dir, "*.png")))

    for i, lc_path in enumerate(lc_tiles):
        reporter.report(f"PROGRESS:Combining tile {i + 1}/{len(lc_tiles)}...")

        filename = os.path.basename(lc_path)
        cd_path = os.path.join(cd_pred_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if not all(os.path.exists(p) for p in [cd_path, mask_path]):
            continue

        lc_array = np.array(Image.open(lc_path).convert("RGB"))
        cd_array = np.array(Image.open(cd_path).convert("RGB"))
        mask_array = np.array(Image.open(mask_path).convert("RGB"))

        final_array = lc_array.copy()
        is_flooded_mask = np.any(cd_array != [0, 0, 0], axis=-1)

        final_array = np.zeros_like(lc_array)
        final_array[is_flooded_mask] = lc_array[is_flooded_mask]

        outside_path_mask = np.all(mask_array == [0, 0, 0], axis=-1)
        final_array[outside_path_mask] = [128, 128, 128]

        Image.fromarray(final_array).save(os.path.join(combined_dir, filename))
    reporter.report("Finished combining tiles.")

    reporter.report("\nStitching combined tiles into a single image...")
    combined_tiles = sorted(glob.glob(os.path.join(combined_dir, "*.png")))
    if not combined_tiles:
        reporter.report("No combined tiles were created. Cannot stitch.")
        return ""
    tile_coords = [
        re.match(r"tile_(\d+)_(\d+)\.png", os.path.basename(p)) for p in combined_tiles
    ]
    valid_tiles = [tc.groups() for tc in tile_coords if tc]
    max_row = max(int(r) for r, c in valid_tiles)
    max_col = max(int(c) for r, c in valid_tiles)

    with Image.open(combined_tiles[0]) as img:
        tile_w, tile_h = img.size

    full_width = (max_col + 1) * tile_w
    full_height = (max_row + 1) * tile_h
    stitched_image = Image.new("RGB", (full_width, full_height))
    reporter.report(f"Creating a {full_width}x{full_height} canvas...")

    for i, tile_path in enumerate(combined_tiles):
        reporter.report(f"PROGRESS:Stitching tile {i + 1}/{len(combined_tiles)}...")

        match = re.match(r"tile_(\d+)_(\d+)\.png", os.path.basename(tile_path))
        if match:
            row, col = map(int, match.groups())
            with Image.open(tile_path) as tile:
                stitched_image.paste(tile, (col * tile_w, row * tile_h))

    reporter.report("Stitching complete.")

    return stitched_image


def save_stitched_image(
    stitched_image: Image.Image, output_path: str, progress_callback=None
) -> str:
    reporter.set_callback(progress_callback)

    if not isinstance(stitched_image, Image.Image):
        reporter.report(
            "Warning: Cannot save stitched map, as the provided input is not a valid image object."
        )
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    stitched_image.save(output_path)
    reporter.report(f"\nFinal output saved to '{output_path}'.")
    return output_path


def save_legend_image(
    legend_image: Image.Image, output_path: str, progress_callback=None
) -> str:
    reporter.set_callback(progress_callback)

    if not isinstance(legend_image, Image.Image):
        reporter.report(
            "Warning: Cannot save legend, as the provided input is not a valid image object."
        )
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    legend_image.save(output_path)
    reporter.report(f"Legend saved to '{output_path}'.")
    return output_path


def calculate_statistics(stitched_image: Image.Image, pixel_size_meters: float) -> dict:
    reporter.report("\nCalculating final statistics...")
    if not isinstance(stitched_image, Image.Image):
        reporter.report(
            "Error: Input is not a valid PIL Image. Cannot calculate stats."
        )
        return {}

    image_array = np.array(stitched_image.convert("RGB"))

    pixel_area_m2 = pixel_size_meters * pixel_size_meters
    m2_to_km2 = 1 / 1_000_000

    grey_color = np.array([128, 128, 128])
    black_color = np.array([0, 0, 0])

    is_grey_mask = np.all(image_array == grey_color, axis=-1)
    flight_path_mask = ~is_grey_mask
    total_path_pixels = np.sum(flight_path_mask)

    is_black_mask = np.all(image_array == black_color, axis=-1)

    non_flooded_pixels = np.sum(is_black_mask & flight_path_mask)
    total_flooded_pixels = total_path_pixels - non_flooded_pixels

    stats = {
        "total_flight_path_area_km2": total_path_pixels * pixel_area_m2 * m2_to_km2,
        "total_flooded_area_km2": total_flooded_pixels * pixel_area_m2 * m2_to_km2,
        "total_non_flooded_area_km2": non_flooded_pixels * pixel_area_m2 * m2_to_km2,
    }

    land_cover_stats = {}
    for class_id, color_rgb in INDEX_TO_COLOR.items():
        class_label = CLASS_ID_TO_LABEL.get(class_id, f"Unknown Class {class_id}")
        color_np = np.array(list(color_rgb))

        class_mask = np.all(image_array == color_np, axis=-1) & flight_path_mask
        pixel_count = np.sum(class_mask)

        area_km2 = pixel_count * pixel_area_m2 * m2_to_km2
        land_cover_stats[class_label] = area_km2

    stats["land_cover_areas_km2"] = land_cover_stats
    reporter.report("Statistics calculation complete.")

    return stats


def generate_legend_image(stats: dict) -> Image.Image:
    reporter.report("\nGenerating legend image...")
    if not stats:
        return None

    total_area = stats.get("total_flight_path_area_km2", 0)
    flooded_area = stats.get("total_flooded_area_km2", 0)
    non_flooded_area = stats.get("total_non_flooded_area_km2", 0)
    land_cover_areas = stats.get("land_cover_areas_km2", {})

    sorted_land_cover = sorted(
        [(k, v) for k, v in land_cover_areas.items() if v > 0.001],
        key=lambda item: item[1],
        reverse=True,
    )

    label_to_id = {v: k for k, v in CLASS_ID_TO_LABEL.items()}

    row_height = 0.35
    header_space = 0.6
    num_items = 3 + len(sorted_land_cover)
    fig_height = (num_items + 2) * row_height
    fig_width = 4.8

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    ax.axis("off")

    y_pos = fig_height - header_space

    def draw_row(label, value=None, color=None, bold=False):
        nonlocal y_pos
        rect_height = 0.25
        if color:
            rect = mpatches.Rectangle(
                (0.2, y_pos - rect_height / 2),
                0.25,
                rect_height,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)
            text_x = 0.55
        else:
            text_x = 0.2

        ax.text(
            text_x,
            y_pos,
            label,
            ha="left",
            va="center",
            fontsize=9,
            weight="bold" if bold else "normal",
        )

        if value is not None:
            ax.text(
                fig_width - 0.2,
                y_pos,
                f"{value:.2f} km²",
                ha="right",
                va="center",
                fontsize=9,
                weight="bold" if bold else "normal",
            )

        y_pos -= row_height

    draw_row("Total Area (Flight Path)", total_area, bold=True)
    draw_row("Total Flooded Area", flooded_area, bold=True)

    draw_row("Non-Flooded Area", non_flooded_area, color="#000000")

    y_pos -= 0.15
    ax.plot([0.2, fig_width - 0.2], [y_pos, y_pos], color="#AAA", linewidth=0.5)
    y_pos -= 0.25

    if sorted_land_cover:
        y_pos -= 0.2
        ax.text(
            fig_width / 2,
            y_pos,
            "Flooded Land Cover Breakdown",
            ha="center",
            va="center",
            fontsize=9,
            style="italic",
            color="#000000",
        )
        y_pos -= row_height

        for label, area in sorted_land_cover:
            class_id = label_to_id.get(label)
            if class_id:
                rgb = INDEX_TO_COLOR.get(class_id)
                if rgb:
                    color_hex = "#%02x%02x%02x" % rgb
                    draw_row(label, area, color=color_hex)

    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0.25,
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    legend_image = Image.open(buf)
    plt.close(fig)

    reporter.report("Legend image generated.")
    return legend_image


def run_flood_mapping_pipeline(
    sar_tif: str,
    optical_tif: str,
    weights_file: str,
    progress_callback=None,
    tile_width: int = 256,
    tile_height: int = 256,
    pixel_size_meters: float = 20.0,
    output_dir: str = None,
) -> tuple:
    reporter.set_callback(progress_callback)

    with tempfile.TemporaryDirectory() as temp_dir:
        reporter.report(f"Created temporary directory: {temp_dir}")

        stitched_image, legend_image = None, None

        sar_tiles_dir, opt_tiles_dir, mask_tiles_dir = tile_sar_and_optical(
            sar_tif,
            optical_tif,
            tile_width,
            tile_height,
            pixel_size_meters,
            base_temp_dir=temp_dir,
            progress_callback=progress_callback,
        )

        lc_pred_dir = classify_landcover(
            input_dir=sar_tiles_dir,
            weights_path=weights_file,
            base_temp_dir=temp_dir,
            progress_callback=progress_callback,
        )

        cd_pred_dir = detect_change_dummy(
            pre_event_dir=opt_tiles_dir,
            post_event_dir=sar_tiles_dir,
            base_temp_dir=temp_dir,
            progress_callback=progress_callback,
        )

        stitched_image = combine_and_stitch_results(
            lc_pred_dir=lc_pred_dir,
            cd_pred_dir=cd_pred_dir,
            mask_dir=mask_tiles_dir,
            base_temp_dir=temp_dir,
            progress_callback=progress_callback,
        )

        if isinstance(stitched_image, Image.Image):
            stats = calculate_statistics(stitched_image, pixel_size_meters)
            if stats:
                legend_image = generate_legend_image(stats)

        if output_dir:
            final_output_path, legend_output_path = None, None
            if stitched_image:
                base_name = (
                    os.path.splitext(os.path.basename(sar_tif))[0] + "_flood_map.png"
                )
                final_output_path = os.path.join(output_dir, base_name)
                save_stitched_image(
                    stitched_image, final_output_path, progress_callback
                )

            if legend_image:
                legend_base_name = (
                    os.path.splitext(os.path.basename(sar_tif))[0] + "_legend.png"
                )
                legend_output_path = os.path.join(output_dir, legend_base_name)
                save_legend_image(legend_image, legend_output_path, progress_callback)

            reporter.report("\nPipeline finished successfully!")
            return (final_output_path, legend_output_path)
        else:
            reporter.report(
                "\nPipeline finished successfully! No output_dir provided, returning image objects."
            )
            return (stitched_image, legend_image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect flooded areas using SAR and optical imagery and classify land cover in the affected regions."
    )
    parser.add_argument("sar_tif", help="Path to the SAR (flight path) .tif")
    parser.add_argument("optical_tif", help="Path to the optical .tif")
    parser.add_argument(
        "weights_file",
        help="Path to the land cover model weights file (e.g., SegformerJaccardLoss.pth)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory to save the final flood map image",
    )
    args = parser.parse_args()

    final_map, legend_map = run_flood_mapping_pipeline(
        sar_tif=args.sar_tif,
        optical_tif=args.optical_tif,
        weights_file=args.weights_file,
        output_dir=args.output_dir,
    )

    if args.output_dir:
        print(f"Outputs saved: Map='{final_map}', Legend='{legend_map}'")
    else:
        if final_map:
            final_map.show()
        if legend_map:
            legend_map.show()
