# Ancient Tamil Character Recognition

This project recognizes ancient Tamil inscription characters using image preprocessing, character region detection, and a ResNet18 classifier.

## Complete Project Flow

1. Prepare labeled character dataset in `labeled_data_final/`.
2. Train classifier using `model_training.py`.
3. Save trained weights as `tamil_inscription_model.pth`.
4. Run one of the preprocessing/detection pipelines:
	- `wavelet.py` (wavelet + CLAHE branch)
	- `test_images.py` (neural cleaning branch)
	- `hybrid_ocr.py` (combined branch, recommended)
5. Predict character labels for extracted regions.
6. Generate annotated result images and recognized text.

## Project Components

### 1) Training (`model_training.py`)

- Loads data from `labeled_data_final/` with grayscale + resize + augmentation.
- Trains `torchvision.models.resnet18` for multi-class Tamil character classification.
- Saves trained model to `tamil_inscription_model.pth`.

Run:

```bash
python model_training.py
```

### 2) Wavelet Detection Branch (`wavelet.py`)

Purpose:
- Strong contour-based localization of character bounding boxes.

Pipeline:
1. Load image from `test_images/`.
2. Convert to grayscale.
3. Apply wavelet denoising (`db2`, level=2).
4. Apply CLAHE enhancement.
5. Binarize (Gaussian blur + Otsu + optional inversion).
6. Morphological opening.
7. Detect contours and filter by width/height/area.
8. Save bounding boxes and cropped character patches.

Default output:
- `wavelet_output/<image_name>/`

Run:

```bash
python wavelet.py
```

### 3) Neural Cleaning Branch (`test_images.py`)

Purpose:
- Produce cleaner character foreground mask before extraction.

Pipeline:
1. Load grayscale image and normalize to `[0,1]`.
2. Apply `StoneEnhancer` CNN (noise estimation and subtraction).
3. Gaussian smoothing.
4. Sauvola thresholding.
5. Morphological cleanup.
6. Connected component filtering.
7. Save cleaned image, optional boxes, and extracted normalized character crops.

Default outputs:
- `test_images_output/`
- `test_images_cleaned/`
- `extracted_characters/`

Run:

```bash
python test_images.py
```


### 4) Hybrid OCR (`hybrid_ocr.py`) - Current End-to-End Pipeline

Purpose:
- Use wavelet branch for stable bounding boxes.
- Use neural branch for cleaner pixels.
- Predict with the trained classifier.
- Annotate predictions on the cleaned image.

Exact flow per input image:
1. Branch A (`wavelet.py` logic): generate wavelet/CLAHE image and detect bounding boxes.
2. Branch B (`test_images.py` logic): generate cleaned binary-like image.
3. Reuse Branch A boxes on Branch B cleaned image.
4. Crop each cleaned character region using wavelet box coordinates.
5. Normalize crop to square (`64x64` intermediate), then transform to model size (`224x224`).
6. Predict Tamil class using `tamil_inscription_model.pth`.
7. Draw green bounding box + predicted label text above the box on cleaned image.
8. Save final annotated image and print recognized sequence.

Output structure:
- `hybrid_ocr_output/<image_name>/wavelet/`
- `hybrid_ocr_output/<image_name>/cleaned/`
- `hybrid_ocr_output/<image_name>/final/<image_name>_annotated_on_cleaned.jpg`

Run:

```bash
python hybrid_ocr.py
```

## Installation

Create and activate virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Important Paths

- Input images: `test_images/`
- Labels dataset: `labeled_data_final/`
- Trained weights: `tamil_inscription_model.pth`

