"""Hybrid OCR pipeline.

This script combines:
- the wavelet preprocessing branch from ``wavelet.py`` to obtain bounding boxes
- the neural-cleaning branch from ``test_images.py`` to obtain a cleaned mask
- the recognition branch from ``ocr.py`` to predict characters

The wavelet boxes are reused as the geometry for character crops, but the
actual character pixels come from the cleaned image produced by the neural
branch.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage import filters, morphology
from skimage.filters import threshold_sauvola
from torchvision import models, transforms


MODEL_PATH = "tamil_inscription_model.pth"
DATASET_PATH = "labeled_data_final"
IMG_SIZE = 224


# ============================================================
# Recognition Model
# ============================================================

class StoneEnhancer(nn.Module):

    def __init__(self):
        super(StoneEnhancer, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):

        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        noise = self.conv3(x2)

        return x - noise


def initialize_weights(model):

    for module in model.modules():

        if isinstance(module, nn.Conv2d):

            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)


def load_recognition_model(model_path=MODEL_PATH, dataset_path=DATASET_PATH):

    class_names = sorted(os.listdir(dataset_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    return model, class_names, device, transform


def predict_char(model, class_names, device, transform, img):

    img_pil = Image.fromarray(img)
    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        _, pred = torch.max(logits, 1)

    return class_names[pred.item()]


# ============================================================
# Wavelet Branch
# ============================================================

def load_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    return img


def wavelet_denoise(gray_img, wavelet="db2", level=2, threshold=20):

    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)
    coeffs_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

    approx_size = coeffs[0].size
    coeffs_thresholded = coeffs_arr.copy()

    detail_coeffs = coeffs_arr.flatten()[approx_size:]
    detail_coeffs_thresholded = pywt.threshold(detail_coeffs, threshold, mode="soft")
    coeffs_thresholded.flatten()[approx_size:] = detail_coeffs_thresholded

    coeffs_from_arr = pywt.array_to_coeffs(
        coeffs_thresholded,
        coeff_slices,
        output_format="wavedec2",
    )

    denoised_img = pywt.waverec2(coeffs_from_arr, wavelet)
    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)

    return denoised_img


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def binarize_image(clahe_image):

    if len(clahe_image.shape) == 3:
        gray = cv2.cvtColor(clahe_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = clahe_image.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)

    if white_pixels > black_pixels:
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, gray


def detect_contours(binary_image):

    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if w >= 10 and h >= 10 and area >= 100:
            valid_contours.append(contour)

    return valid_contours


def wavelet_boxes_from_image(image_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    original_img = load_image(image_path)
    cv2.imwrite(os.path.join(output_dir, "01_original.jpg"), original_img)

    grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "02_grayscale.jpg"), grayscale_img)

    wavelet_denoised = wavelet_denoise(grayscale_img)
    cv2.imwrite(os.path.join(output_dir, "03_wavelet.jpg"), wavelet_denoised)

    clahe_img = apply_clahe(wavelet_denoised)
    cv2.imwrite(os.path.join(output_dir, "04_clahe.jpg"), clahe_img)

    binary, _ = binarize_image(clahe_img)
    cv2.imwrite(os.path.join(output_dir, "05_binary_for_detection.jpg"), binary)

    contours = detect_contours(binary)

    boxes = []
    for contour in contours:
        boxes.append(cv2.boundingRect(contour))

    boxes = sorted(boxes, key=lambda box: (box[1] // 50, box[0]))

    box_vis = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in boxes:
        cv2.rectangle(box_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_dir, "06_wavelet_boxes.jpg"), box_vis)

    return boxes


# ============================================================
# Neural Cleaning Branch
# ============================================================

def normalize_character(char_img, size=64):

    h, w = char_img.shape
    max_dim = max(h, w)

    canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)

    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2

    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = char_img

    char_pil = Image.fromarray(canvas)
    char_pil = char_pil.resize((size, size), Image.BILINEAR)

    return np.array(char_pil)


def cleaned_image_from_test_branch(image_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    image_name = Path(image_path).stem

    image = Image.open(image_path).convert("L")
    image_np = np.array(image) / 255.0

    Image.fromarray((image_np * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "step_01_original.jpg")
    )

    tensor_img = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float()

    enhancer = StoneEnhancer()
    initialize_weights(enhancer)
    enhancer.eval()

    with torch.no_grad():
        enhanced = enhancer(tensor_img)

    enhanced_np = enhanced.squeeze().numpy()
    enhanced_np = np.clip(enhanced_np, 0, 1)

    Image.fromarray((enhanced_np * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "step_03_nn_enhanced.jpg")
    )

    smoothed = filters.gaussian(enhanced_np, sigma=1.5)

    Image.fromarray((smoothed * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "step_04_smoothed.jpg")
    )

    thresh = threshold_sauvola(smoothed, window_size=75, k=0.15)
    binary = smoothed < thresh

    Image.fromarray((binary * 255).astype(np.uint8)).save(
        os.path.join(output_dir, "step_05_binary.jpg")
    )

    cleaned = morphology.remove_small_objects(binary, min_size=60)
    cleaned = morphology.closing(cleaned, morphology.disk(1))

    cleaned_img = (cleaned * 255).astype(np.uint8)

    Image.fromarray(cleaned_img).save(
        os.path.join(output_dir, "step_06_cleaned.jpg")
    )

    Image.fromarray(cleaned_img).save(
        os.path.join(output_dir, f"{image_name}_cleaned.png")
    )

    return cleaned_img


# ============================================================
# Hybrid OCR Pipeline
# ============================================================

def crop_cleaned_character(cleaned_img, box, padding=4):

    x, y, w, h = box

    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(cleaned_img.shape[1], x + w + padding)
    y_end = min(cleaned_img.shape[0], y + h + padding)

    char_crop = cleaned_img[y_start:y_end, x_start:x_end]

    if char_crop.size == 0:
        return None

    return normalize_character(char_crop, 64)


def annotate_predictions(base_image, results, output_path):

    if len(base_image.shape) == 2:
        base_bgr = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base_image.copy()

    pil_img = Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc", 22)
    except Exception:
        font = ImageFont.load_default()

    for x, y, w, h, pred in results:

        draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)

        text_y = max(0, y - 26)
        draw.text((x, text_y), pred, font=font, fill=(255, 0, 0))

    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    saved = cv2.imwrite(str(output_path), result_img)

    if not saved:
        raise IOError(f"Failed to save output image: {output_path}")


def run_hybrid_ocr(image_path, output_root="hybrid_ocr_output"):

    image_path = str(Path(image_path))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model, class_names, device, transform = load_recognition_model()

    image_name = Path(image_path).stem
    image_output_dir = Path(output_root) / image_name
    wavelet_dir = image_output_dir / "wavelet"
    cleaned_dir = image_output_dir / "cleaned"
    final_dir = image_output_dir / "final"

    wavelet_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    boxes = wavelet_boxes_from_image(image_path, str(wavelet_dir))
    cleaned_img = cleaned_image_from_test_branch(image_path, str(cleaned_dir))

    results = []

    for box in boxes:

        char_img = crop_cleaned_character(cleaned_img, box)

        if char_img is None:
            continue

        prediction = predict_char(model, class_names, device, transform, char_img)
        results.append((*box, prediction))

    result_path = final_dir / f"{image_name}_annotated_on_cleaned.jpg"
    annotate_predictions(cleaned_img, results, result_path)

    recognized_text = "".join(prediction for _, _, _, _, prediction in results)

    print(f"Saved annotated image: {result_path}")
    print(f"Predicted text: {recognized_text}")

    return result_path, recognized_text


def run_folder(input_path):
    output_root="hybrid_ocr_output"
    run_hybrid_ocr(str(input_path), output_root=output_root)


if __name__ == "__main__":
    run_folder(r"C:\Users\Lenovo\Documents\Sample\temple_ocr_project\test_images\51.jpg.jpeg")