# Ancient Tamil Character Recognition

This project contains image preprocessing and character extraction pipelines for ancient Tamil inscription images.

## Scripts

### `wavelet.py`

Combined pipeline that does:

1. Preprocessing:
- Load image
- Grayscale conversion
- Wavelet denoising
- CLAHE enhancement

2. Character detection:
- Binarization (Otsu + morphology)
- Contour filtering
- Green bounding boxes
- Character crop extraction

Input folder:
- `test_images/`

Output folder:
- `wavelet_output/`

Per image output (inside `wavelet_output/<image_name>/`):
- `01_original.jpg`
- `02_grayscale.jpg`
- `03_wavelet.jpg`
- `04_clahe.jpg`
- `binary_for_detection.jpg`
- `clahe_bounding_boxes.jpg`
- `characters/char_*.png`

Run:

```bash
python wavelet.py
```

### `test_images.py`

Neural-enhancement based OCR preprocessing pipeline that:

- Reads images from `test_images/`
- Produces cleaned images
- Draws bounding boxes
- Extracts normalized character crops

Default outputs:
- `test_images_output/`
- `test_images_cleaned/`
- `extracted_characters/`

Run:

```bash
python test_images.py
```

## Neural Enhancement Pipeline (`test_images.py`)

Processing flow per image:

1. Load image in grayscale and normalize to `[0, 1]`.
2. Pass through `StoneEnhancer` CNN (`conv -> relu -> conv -> relu -> conv`) to estimate and subtract noise.
3. Save neural-enhanced result: `step_03_nn_enhanced.jpg`.
4. Apply Gaussian smoothing (`sigma=1.5`) and save: `step_04_smoothed.jpg`.
5. Apply Sauvola thresholding (`window_size=75`, `k=0.15`) and save binary: `step_05_binary.jpg`.
6. Remove small objects and apply morphological closing, then save: `step_06_cleaned.jpg`.
7. Run connected-component analysis and region filtering (area, size, aspect ratio).
8. Draw bounding boxes and save: `step_07_bounding_boxes.jpg`.
9. Crop each detected region, normalize each character to `64x64`, and save to `extracted_characters/`.

Additional cleaned-image dataset output:

- `test_images_cleaned/<image_name>_cleaned.png`

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```


## Notes

- `wavelet.py` recreates `wavelet_output/` on each run.
- Bounding boxes in `wavelet.py` are drawn in green.
