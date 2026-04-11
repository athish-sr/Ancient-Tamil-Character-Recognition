"""
Binarize extracted characters (no stroke thickening) then cluster them.

Binarization pipeline:
  1. Mild Gaussian blur → Otsu threshold → ink = white
  2. Connected component filtering: drop components < 8% of largest
     (removes noise specks, keeps all real stroke parts)
  3. Tiny morphological open (2x2) — kills remaining 1-2px salt noise
  4. Center on 64x64 canvas (aspect-ratio preserved, NO dilation/closing)

Clustering:
  HOG + Hu moments → StandardScaler → PCA (95%) → DBSCAN (eps=17)
"""

import cv2
import numpy as np
import os
import shutil
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

CHAR_SIZE   = 64
SRC_DIR     = "all_characters"
BINARY_DIR  = "all_characters_binary"
CLUSTER_DIR = "clusters_binary"


# ── binarize ──────────────────────────────────────────────────────────────────

def binarize(img_gray):
    """
    Otsu binarize → CC noise removal → tiny open → center on canvas.
    No morphological closing — strokes are NOT thickened.
    """
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ensure ink = white
    if np.sum(bw == 255) > np.sum(bw == 0):
        bw = cv2.bitwise_not(bw)

    # connected component noise removal
    n_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n_labels > 1:
        areas    = stats[1:, cv2.CC_STAT_AREA]
        max_area = int(areas.max())
        min_keep = max(20, int(max_area * 0.08))   # keep >= 8% of largest
        clean = np.zeros_like(bw)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_keep:
                clean[labels_map == lbl] = 255
        bw = clean

    # remove remaining 1-2px salt specks
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k2, iterations=1)

    # tight crop to ink
    coords = cv2.findNonZero(bw)
    if coords is None:
        return np.zeros((CHAR_SIZE, CHAR_SIZE), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = bw[y:y+h, x:x+w]

    # scale to canvas preserving aspect ratio
    pad    = 6
    target = CHAR_SIZE - 2 * pad
    scale  = target / max(w, h, 1)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((CHAR_SIZE, CHAR_SIZE), dtype=np.uint8)
    yo = (CHAR_SIZE - nh) // 2
    xo = (CHAR_SIZE - nw) // 2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas


# ── features ──────────────────────────────────────────────────────────────────

def extract_features(bw):
    hog_feat = hog(bw, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    hu = cv2.HuMoments(cv2.moments(bw)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return np.concatenate([hog_feat, hu * 10])


# ── contact sheet ─────────────────────────────────────────────────────────────

def make_sheet(folder, out_path, cols=10, thumb=64):
    imgs = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
    if not imgs:
        return
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.ones((rows * thumb, cols * thumb), dtype=np.uint8) * 200
    for idx, fn in enumerate(imgs):
        im = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (thumb, thumb))
        r, c = divmod(idx, cols)
        sheet[r*thumb:(r+1)*thumb, c*thumb:(c+1)*thumb] = im
    cv2.imwrite(out_path, sheet)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── step 1: binarize ─────────────────────────────────────────────────────
    print("=" * 55)
    print("STEP 1: BINARIZE")
    print("=" * 55)

    if os.path.exists(BINARY_DIR):
        shutil.rmtree(BINARY_DIR)
    os.makedirs(BINARY_DIR)

    src_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith('.png')])
    features, valid_paths = [], []

    for fname in src_files:
        img = cv2.imread(os.path.join(SRC_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        bw  = binarize(img)
        cv2.imwrite(os.path.join(BINARY_DIR, fname), bw)
        features.append(extract_features(bw))
        valid_paths.append(os.path.join(BINARY_DIR, fname))

    print(f"✓ {len(valid_paths)} characters binarized → {BINARY_DIR}/")

    # ── step 2: cluster ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("STEP 2: CLUSTER")
    print("=" * 55)

    features = np.array(features)
    print(f"Feature shape: {features.shape}")

    fs = StandardScaler().fit_transform(features)

    n_comp = min(80, len(fs) - 1, fs.shape[1])
    pca    = PCA(n_components=n_comp, random_state=42)
    fp     = pca.fit_transform(fs)
    n_95   = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95)) + 1
    fp     = fp[:, :n_95]
    print(f"✓ PCA: {n_95} components (95% variance)")

    db     = DBSCAN(eps=17, min_samples=2, metric='euclidean', n_jobs=-1)
    labels = db.fit_predict(fp)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int(np.sum(labels == -1))
    print(f"✓ DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    # assign noise to nearest centroid
    if n_noise > 0 and n_clusters > 0:
        cluster_ids = [l for l in set(labels) if l != -1]
        centroids   = {cid: fp[labels == cid].mean(axis=0) for cid in cluster_ids}
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = min(centroids, key=lambda cid: np.linalg.norm(fp[i] - centroids[cid]))
        print(f"✓ Noise assigned to nearest cluster")

    # save
    if os.path.exists(CLUSTER_DIR):
        shutil.rmtree(CLUSTER_DIR)
    os.makedirs(CLUSTER_DIR)

    cluster_counts = {}
    for path, label in zip(valid_paths, labels):
        folder = os.path.join(CLUSTER_DIR, f"cluster_{int(label):03d}")
        os.makedirs(folder, exist_ok=True)
        shutil.copy(path, os.path.join(folder, os.path.basename(path)))
        cluster_counts[int(label)] = cluster_counts.get(int(label), 0) + 1

    for k in sorted(cluster_counts):
        folder = os.path.join(CLUSTER_DIR, f"cluster_{k:03d}")
        make_sheet(folder, os.path.join(CLUSTER_DIR, f"cluster_{k:03d}_sheet.jpg"))

    sizes = sorted(cluster_counts.values(), reverse=True)
    print(f"\n✓ {len(valid_paths)} characters → {len(cluster_counts)} clusters")
    print(f"  size: min={min(sizes)}, max={max(sizes)}, avg={int(sum(sizes)/len(sizes))}")
    print(f"  saved in: {CLUSTER_DIR}/")


if __name__ == "__main__":
    main()
