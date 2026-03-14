"""
FULL PIPELINE:
Stone Inscription Preprocessing + Character Detection

All outputs are saved in ONE folder per image.
"""

import cv2
import numpy as np
import pywt
import os
import shutil


# ============================================================
# ---------------------- W1 FUNCTIONS ------------------------
# ============================================================

def load_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    print(f"✓ Image loaded: {img.shape}")

    return img


def wavelet_denoise(gray_img, wavelet='db2', level=2, threshold=20):

    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)

    coeffs_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    approx_size = coeffs[0].size

    coeffs_thresholded = coeffs_arr.copy()

    detail_coeffs = coeffs_arr.flatten()[approx_size:]

    detail_coeffs_thresholded = pywt.threshold(detail_coeffs, threshold, mode='soft')

    coeffs_thresholded.flatten()[approx_size:] = detail_coeffs_thresholded

    coeffs_from_arr = pywt.array_to_coeffs(
        coeffs_thresholded,
        coeff_slices,
        output_format='wavedec2'
    )

    denoised_img = pywt.waverec2(coeffs_from_arr, wavelet)

    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)

    print("✓ Wavelet denoising applied")

    return denoised_img


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_img = clahe.apply(img)

    print("✓ CLAHE enhancement applied")

    return enhanced_img


def process_single_image(image_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    original_img = load_image(image_path)
    cv2.imwrite(os.path.join(output_dir, '01_original.jpg'), original_img)

    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, '02_grayscale.jpg'), grayscale_img)

    print("✓ Grayscale conversion complete")

    wavelet_denoised = wavelet_denoise(grayscale_img)

    cv2.imwrite(os.path.join(output_dir, '03_wavelet.jpg'), wavelet_denoised)

    clahe_img = apply_clahe(wavelet_denoised)

    cv2.imwrite(os.path.join(output_dir, '04_clahe.jpg'), clahe_img)

    return clahe_img


# ============================================================
# ---------------------- W2 FUNCTIONS ------------------------
# ============================================================

def binarize_image(clahe_image):

    if len(clahe_image.shape) == 3:
        gray = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY)
        print("✓ Converted to grayscale")
    else:
        gray = clahe_image.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    print("✓ Gaussian blur applied")

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print("✓ Otsu threshold applied")

    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)
        print("✓ Image inverted")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    print("✓ Morphological opening applied")

    return binary, gray


def detect_contours(binary_image):

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"✓ Found {len(contours)} contours")

    valid_contours = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if w >= 10 and h >= 10 and area >= 100:
            valid_contours.append(contour)

    print(f"✓ Filtered to {len(valid_contours)} character contours")

    return valid_contours


def draw_bounding_boxes(clahe_image, contours, output_path):

    if len(clahe_image.shape) == 2:
        clahe_with_boxes = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    else:
        clahe_with_boxes = clahe_image.copy()
        
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(clahe_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    cv2.imwrite(output_path, clahe_with_boxes)

    print("✓ Bounding boxes saved")


def save_characters(clahe_image, contours, char_dir):

    os.makedirs(char_dir, exist_ok=True)

    if len(clahe_image.shape) == 3:
        clahe_gray = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY)
    else:
        clahe_gray = clahe_image.copy()

    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    sorted_indices = sorted(
        range(len(bounding_boxes)),
        key=lambda i: (bounding_boxes[i][1] // 50, bounding_boxes[i][0])
    )

    for idx, contour_idx in enumerate(sorted_indices):

        x, y, w, h = bounding_boxes[contour_idx]

        padding = 5

        x_start = max(0, x - padding)
        y_start = max(0, y - padding)

        x_end = min(clahe_gray.shape[1], x + w + padding)
        y_end = min(clahe_gray.shape[0], y + h + padding)

        char_img = clahe_gray[y_start:y_end, x_start:x_end]

        cv2.imwrite(os.path.join(char_dir, f'char_{idx}.png'), char_img)

    print(f"✓ Saved {len(sorted_indices)} characters")


# ============================================================
# ------------------------ MAIN ------------------------------
# ============================================================

def main():

    INPUT_FOLDER = "test_images"
    OUTPUT_FOLDER = "wavelet_output"

    exts = ('.jpg', '.jpeg', '.png')

    images = sorted(
        [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(exts)]
    )

    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)

    os.makedirs(OUTPUT_FOLDER)

    print("="*70)
    print("STONE INSCRIPTION FULL PIPELINE")
    print("="*70)

    total_chars = 0

    for i, fname in enumerate(images, start=1):

        print(f"\n[{i}/{len(images)}] Processing {fname}")

        image_path = os.path.join(INPUT_FOLDER, fname)

        name = os.path.splitext(fname)[0]

        out_dir = os.path.join(OUTPUT_FOLDER, name)

        os.makedirs(out_dir)

        try:

            clahe_img = process_single_image(image_path, out_dir)

            binary, _ = binarize_image(clahe_img)

            cv2.imwrite(os.path.join(out_dir, "binary_for_detection.jpg"), binary)

            contours = detect_contours(binary)

            draw_bounding_boxes(
                clahe_img,
                contours,
                os.path.join(out_dir, "clahe_bounding_boxes.jpg")
            )

            char_dir = os.path.join(out_dir, "characters")

            save_characters(clahe_img, contours, char_dir)

            total_chars += len(contours)

        except Exception as e:

            print(f"✗ Failed: {e}")

    print("\n"+"="*70)
    print("PIPELINE COMPLETE")
    print("="*70)

    print(f"Total characters detected: {total_chars}")
    print(f"Results saved in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()