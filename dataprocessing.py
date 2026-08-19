"""Build an HDF5 training set from aligned infrared/visible image pairs."""

import argparse
from pathlib import Path

import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm


IMAGE_SUFFIXES = {".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff", ".npy"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare paired image patches for PromptRoute-Mamba.")
    parser.add_argument("--ir-dir", type=Path, required=True, help="Directory containing infrared images.")
    parser.add_argument("--vi-dir", type=Path, required=True, help="Directory containing visible images.")
    parser.add_argument("--output", type=Path, default=Path("data/MSRS_train_imgsize_256_stride_100.h5"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=100)
    return parser.parse_args()


def image_files(folder):
    return {path.name: path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}


def rgb_to_y(image):
    if image.ndim == 2:
        return image[None, ...]
    image = image.transpose(2, 0, 1)
    return image[0:1] * 0.299 + image[1:2] * 0.587 + image[2:3] * 0.114


def image_to_patches(image, window, stride):
    channels, height, width = image.shape
    rows = range(0, height - window + 1, stride)
    cols = range(0, width - window + 1, stride)
    patches = [image[:, row : row + window, col : col + window] for row in rows for col in cols]
    if not patches:
        return np.empty((channels, window, window, 0), dtype=np.float32)
    return np.stack(patches, axis=-1).astype(np.float32)


def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10, upper_percentile=90):
    lower, upper = np.percentile(image, [lower_percentile, upper_percentile])
    return upper == 0 or (upper - lower) / upper < fraction_threshold


def read_unit_image(path):
    return imread(path).astype(np.float32) / 255.0


def main():
    args = parse_args()
    ir_files = image_files(args.ir_dir)
    vi_files = image_files(args.vi_dir)
    names = sorted(ir_files.keys() & vi_files.keys())
    if not names:
        raise RuntimeError("No aligned image pairs with matching filenames were found.")

    missing_ir = sorted(vi_files.keys() - ir_files.keys())
    missing_vi = sorted(ir_files.keys() - vi_files.keys())
    if missing_ir or missing_vi:
        raise RuntimeError(f"Unpaired files detected: {len(missing_ir)} missing IR, {len(missing_vi)} missing visible.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    patch_count = 0
    with h5py.File(args.output, "w") as h5_file:
        h5_ir = h5_file.create_group("ir_patchs")
        h5_vi = h5_file.create_group("vis_patchs")
        for name in tqdm(names, desc="Preparing patches"):
            ir = read_unit_image(ir_files[name])
            ir = ir[None, ...] if ir.ndim == 2 else ir.transpose(2, 0, 1)[0:1]
            vi = rgb_to_y(read_unit_image(vi_files[name]))
            ir_patches = image_to_patches(ir, args.patch_size, args.stride)
            vi_patches = image_to_patches(vi, args.patch_size, args.stride)
            for index in range(ir_patches.shape[-1]):
                ir_patch = ir_patches[0, :, :, index]
                vi_patch = vi_patches[0, :, :, index]
                if is_low_contrast(ir_patch) or is_low_contrast(vi_patch):
                    continue
                h5_ir.create_dataset(str(patch_count), data=ir_patch[None, ...])
                h5_vi.create_dataset(str(patch_count), data=vi_patch[None, ...])
                patch_count += 1
    print(f"Saved {patch_count} aligned patches to {args.output}")


if __name__ == "__main__":
    main()
