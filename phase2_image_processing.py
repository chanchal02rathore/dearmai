import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE_DIR      = r"C:\Users\CHANCHAL\Desktop\HAM10000"
PROCESSED_DIR = os.path.join(BASE_DIR, "processed", "images")
CSV_PATH      = os.path.join(BASE_DIR, "processed", "metadata_clean.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "processed", "final_images")
IMG_SIZE      = 224

os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "val"),   exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "test"),  exist_ok=True)

# ─────────────────────────────────────────
# STEP 1: Load clean CSV
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading clean metadata...")
df = pd.read_csv(CSV_PATH)
print(f"Total images: {len(df)}")

# ─────────────────────────────────────────
# STEP 2: Hair Removal Function
# ─────────────────────────────────────────
def remove_hair(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blackhat filter — detects dark hair on bright background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold — hair pixels ko mask karo
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpainting — hair pixels ko surrounding skin se fill karo
    cleaned = cv2.inpaint(image, hair_mask, inpaintRadius=3,
                          flags=cv2.INPAINT_TELEA)
    return cleaned

# ─────────────────────────────────────────
# STEP 3: Process + Save All Images
# ─────────────────────────────────────────
print("\nSTEP 2 & 3: Hair removal + Resize + Normalize...")
print("(This will take a few minutes...)")

failed = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
    src_path = os.path.join(PROCESSED_DIR, row['image_id'] + ".jpg")
    dst_path = os.path.join(OUTPUT_DIR, row['image_id'] + ".jpg")

    if os.path.exists(dst_path):
        continue

    try:
        # Load image
        img = cv2.imread(src_path)
        if img is None:
            failed.append(row['image_id'])
            continue

        # Hair removal
        img = remove_hair(img)

        # Resize to 224x224
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Save processed image
        cv2.imwrite(dst_path, img)

    except Exception as e:
        failed.append(row['image_id'])

print(f"Processed successfully : {len(df) - len(failed)}")
print(f"Failed                 : {len(failed)}")

# ─────────────────────────────────────────
# STEP 4: Train / Val / Test Split
# ─────────────────────────────────────────
print("\nSTEP 4: Train / Val / Test split...")

# First split — 70% train, 30% temp
train_df, temp_df = train_test_split(
    df, test_size=0.30,
    random_state=42,
    stratify=df['label']   # class balance maintain karo
)

# Second split — 15% val, 15% test
val_df, test_df = train_test_split(
    temp_df, test_size=0.50,
    random_state=42,
    stratify=temp_df['label']
)

print(f"Train : {len(train_df)} images")
print(f"Val   : {len(val_df)} images")
print(f"Test  : {len(test_df)} images")

# Save split CSVs
train_df.to_csv(os.path.join(BASE_DIR, "processed", "train.csv"), index=False)
val_df.to_csv(os.path.join(BASE_DIR, "processed", "val.csv"),   index=False)
test_df.to_csv(os.path.join(BASE_DIR, "processed", "test.csv"), index=False)
print("Split CSVs saved!")

# ─────────────────────────────────────────
# STEP 5: Visualize — Before vs After
# ─────────────────────────────────────────
print("\nSTEP 5: Generating before/after comparison...")

sample_ids = df['image_id'].sample(3, random_state=42).values
fig, axes = plt.subplots(3, 2, figsize=(10, 12))

for i, img_id in enumerate(sample_ids):
    # Original
    orig = cv2.imread(os.path.join(PROCESSED_DIR, img_id + ".jpg"))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # Processed
    proc = cv2.imread(os.path.join(OUTPUT_DIR, img_id + ".jpg"))
    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

    axes[i][0].imshow(orig)
    axes[i][0].set_title(f"Original — {img_id}")
    axes[i][0].axis('off')

    axes[i][1].imshow(proc)
    axes[i][1].set_title(f"Processed (Hair removed + Resized)")
    axes[i][1].axis('off')

plt.tight_layout()
chart_path = os.path.join(BASE_DIR, "processed", "before_after.png")
plt.savefig(chart_path, dpi=150)
plt.show()
print(f"Chart saved: {chart_path}")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 2 COMPLETE — SUMMARY")
print("=" * 50)
print(f"Total processed : {len(df) - len(failed)}")
print(f"Train split     : {len(train_df)}")
print(f"Val split       : {len(val_df)}")
print(f"Test split      : {len(test_df)}")
print(f"Output folder   : {OUTPUT_DIR}")
print("=" * 50)
print("Phase 2 complete! Ready for Phase 3 — CNN Training.")