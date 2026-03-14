import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from skimage import filters, morphology, measure
from skimage.filters import threshold_sauvola
from scipy import ndimage


# ============================================================
# Neural Network
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

    for m in model.modules():

        if isinstance(m, nn.Conv2d):

            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ============================================================
# Character Normalization
# ============================================================

def normalize_character(char_img, size=64):

    h, w = char_img.shape
    max_dim = max(h, w)

    canvas = np.zeros((max_dim, max_dim), dtype=np.uint8)

    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2

    canvas[y_offset:y_offset+h, x_offset:x_offset+w] = char_img

    char_pil = Image.fromarray(canvas)
    char_pil = char_pil.resize((size, size), Image.BILINEAR)

    return np.array(char_pil)


# ============================================================
# Processing Pipeline
# ============================================================

def process_image(image_path,
                  output_dir,
                  cleaned_dataset_dir,
                  char_dataset_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cleaned_dataset_dir, exist_ok=True)
    os.makedirs(char_dataset_dir, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    print("Processing:", image_name)

    image = Image.open(image_path).convert("L")
    image_np = np.array(image) / 255.0

    Image.fromarray((image_np * 255).astype(np.uint8)).save(
        f"{output_dir}/step_01_original.jpg"
    )

    tensor_img = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).float()

    model = StoneEnhancer()
    initialize_weights(model)
    model.eval()

    with torch.no_grad():
        enhanced = model(tensor_img)

    enhanced_np = enhanced.squeeze().numpy()
    enhanced_np = np.clip(enhanced_np, 0, 1)

    Image.fromarray((enhanced_np * 255).astype(np.uint8)).save(
        f"{output_dir}/step_03_nn_enhanced.jpg"
    )

    smoothed = filters.gaussian(enhanced_np, sigma=1.5)

    Image.fromarray((smoothed * 255).astype(np.uint8)).save(
        f"{output_dir}/step_04_smoothed.jpg"
    )

    thresh = threshold_sauvola(smoothed, window_size=75, k=0.15)

    binary = smoothed < thresh

    Image.fromarray((binary * 255).astype(np.uint8)).save(
        f"{output_dir}/step_05_binary.jpg"
    )

    cleaned = morphology.remove_small_objects(binary, min_size=60)
    cleaned = morphology.closing(cleaned, morphology.disk(1))

    cleaned_img = (cleaned * 255).astype(np.uint8)

    Image.fromarray(cleaned_img).save(
        f"{output_dir}/step_06_cleaned.jpg"
    )

    Image.fromarray(cleaned_img).save(
        f"{cleaned_dataset_dir}/{image_name}_cleaned.png"
    )

    # =====================================================
    # Bounding Boxes
    # =====================================================

    labeled, num_features = ndimage.label(cleaned)
    regions = measure.regionprops(labeled)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cleaned, cmap="gray")

    char_count = 0

    areas = [r.area for r in regions]

    if areas:
        median_area = np.median(areas)
        min_area = max(80, median_area * 0.05)
        max_area = median_area * 20
    else:
        min_area, max_area = 80, 1e9

    for region in regions:

        if region.area < min_area or region.area > max_area:
            continue

        minr, minc, maxr, maxc = region.bbox

        # =================================================
        # Add margin padding
        # =================================================

        padding = 6

        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(cleaned.shape[0], maxr + padding)
        maxc = min(cleaned.shape[1], maxc + padding)

        h = maxr - minr
        w = maxc - minc

        if w > 200 or h > 150:
            continue

        aspect = w / (h + 1e-6)

        if aspect > 8 or aspect < 0.1:
            continue

        rect = plt.Rectangle(
            (minc, minr),
            w,
            h,
            fill=False,
            edgecolor="red",
            linewidth=1.5
        )

        ax.add_patch(rect)

        # =================================================
        # Character Extraction
        # =================================================

        char_crop = cleaned[minr:maxr, minc:maxc]
        char_img = (char_crop * 255).astype(np.uint8)

        char_img = normalize_character(char_img, 64)

        char_filename = f"{image_name}_char_{char_count:04d}.png"

        Image.fromarray(char_img).save(
            os.path.join(char_dataset_dir, char_filename)
        )

        char_count += 1

    plt.axis("off")

    plt.savefig(
        f"{output_dir}/step_07_bounding_boxes.jpg",
        bbox_inches="tight",
        dpi=150
    )

    plt.close()

    print("Characters detected:", char_count)


# ============================================================
# Batch Processing
# ============================================================

def run_pipeline(input_folder="test_images",
                 output_folder="test_images_output",
                 cleaned_dataset_dir="test_images_cleaned",
                 char_dataset_dir="extracted_characters"):

    os.makedirs(cleaned_dataset_dir, exist_ok=True)
    os.makedirs(char_dataset_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.png")) + \
                  glob.glob(os.path.join(input_folder, "*.jpeg"))

    if not image_files:
        print("No images found")
        return

    for image_path in image_files:

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        output_dir = os.path.join(output_folder, image_name)

        process_image(
            image_path,
            output_dir,
            cleaned_dataset_dir,
            char_dataset_dir
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_pipeline()