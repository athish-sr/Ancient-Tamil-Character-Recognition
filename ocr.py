import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import measure

MODEL_PATH = "tamil_inscription_model.pth"
DATASET_PATH = "labeled_data_final"
IMG_SIZE = 224

# -----------------------------
# Load Classes
# -----------------------------
class_names = sorted(os.listdir(DATASET_PATH))
num_classes = len(class_names)


# -----------------------------
# Load Model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✓ Model Loaded")


# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])


# -----------------------------
# IoU Calculation
# -----------------------------
def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])

    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interW = max(0, xB-xA)
    interH = max(0, yB-yA)

    interArea = interW*interH

    areaA = boxA[2]*boxA[3]
    areaB = boxB[2]*boxB[3]

    union = areaA+areaB-interArea

    if union == 0:
        return 0

    return interArea/union


# -----------------------------
# Merge Nearby Boxes
# -----------------------------
def merge_boxes(boxes, iou_thresh=0.25, dist_thresh=15):

    merged = True

    while merged:

        merged = False
        new_boxes = []
        skip = set()

        for i in range(len(boxes)):

            if i in skip:
                continue

            x1,y1,w1,h1 = boxes[i]
            boxA = boxes[i]

            for j in range(i+1,len(boxes)):

                if j in skip:
                    continue

                boxB = boxes[j]

                if iou(boxA,boxB) > iou_thresh:

                    x2,y2,w2,h2 = boxB

                    x_min = min(x1,x2)
                    y_min = min(y1,y2)
                    x_max = max(x1+w1, x2+w2)
                    y_max = max(y1+h1, y2+h2)

                    boxA = (x_min,y_min,x_max-x_min,y_max-y_min)

                    skip.add(j)
                    merged = True

                else:

                    # Distance merging
                    cx1 = x1+w1/2
                    cy1 = y1+h1/2

                    x2,y2,w2,h2 = boxB
                    cx2 = x2+w2/2
                    cy2 = y2+h2/2

                    dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)

                    if dist < dist_thresh:

                        x_min = min(x1,x2)
                        y_min = min(y1,y2)
                        x_max = max(x1+w1, x2+w2)
                        y_max = max(y1+h1, y2+h2)

                        boxA = (x_min,y_min,x_max-x_min,y_max-y_min)

                        skip.add(j)
                        merged = True

            new_boxes.append(boxA)

        boxes = new_boxes

    return boxes


# -----------------------------
# Bounding Box Detection
# -----------------------------
def get_boxes(binary):

    labeled, num = ndimage.label(binary)

    regions = measure.regionprops(labeled)

    boxes=[]

    areas=[r.area for r in regions]

    if areas:
        median=np.median(areas)
        min_area=max(60, median*0.05)
        max_area=median*20
    else:
        min_area,max_area=60,1e9

    for r in regions:

        if r.area < min_area or r.area > max_area:
            continue

        minr,minc,maxr,maxc=r.bbox

        w=maxc-minc
        h=maxr-minr

        if w<10 or h<10:
            continue

        if w>200 or h>200:
            continue

        boxes.append((minc,minr,w,h))

    return boxes


# -----------------------------
# Predict Character
# -----------------------------
def predict_char(img):

    img_pil = Image.fromarray(img)

    tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        _,pred = torch.max(out,1)

    return class_names[pred.item()]


# -----------------------------
# OCR Pipeline
# -----------------------------
def run_ocr(image_path):

    image_path = str(Path(image_path))

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    img=cv2.imread(image_path)

    if img is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    input_path = Path(image_path)
    base = input_path.stem

    out_dir="ocr_output"

    os.makedirs(out_dir,exist_ok=True)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    boxes=get_boxes(binary)

    print("Initial boxes:",len(boxes))

    boxes=merge_boxes(boxes)

    print("Merged boxes:",len(boxes))

    boxes=sorted(boxes,key=lambda b:(b[1]//50,b[0]))

    results=[]

    for i,(x,y,w,h) in enumerate(boxes):

        char=binary[y:y+h,x:x+w]

        if char.size==0:
            continue

        char=cv2.resize(char,(IMG_SIZE,IMG_SIZE))

        pred=predict_char(char)

        results.append((x,y,w,h,pred))


    # Draw results
    pil_img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(pil_img)

    try:
        font=ImageFont.truetype("C:/Windows/Fonts/Nirmala.ttc",22)
    except:
        font=ImageFont.load_default()

    for x,y,w,h,p in results:

        draw.rectangle([x,y,x+w,y+h],outline=(0,255,0),width=2)
        draw.text((x,y-25),p,font=font,fill=(255,0,0))

    result_img=cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)

    result_path = Path(out_dir) / f"{base}_result.jpg"
    saved = cv2.imwrite(str(result_path), result_img)

    if not saved:
        raise IOError(f"Failed to save output image: {result_path}")

    text="".join(p for _,_,_,_,p in results)

    print("\nTamil Text:\n",text)
    print(f"Saved output image: {result_path}")


# -----------------------------
# Run
# -----------------------------
run_ocr(r"test_images_cleaned\51.jpg_cleaned.png")