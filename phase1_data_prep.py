import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
RAW_DIR = r"C:\Users\CHANCHAL\Desktop\HAM10000"

CSV_PATH = os.path.join(RAW_DIR, "HAM10000_metadata.csv")
IMAGES_DIR = os.path.join(RAW_DIR, "images")
df = pd.read_csv(CSV_PATH)
PROCESSED_DIR = os.path.join(RAW_DIR, "processed", "images")


# ─────────────────────────────────────────
# STEP 1: Load metadata
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading metadata...")
print(f"Total records : {len(df)}")
print(f"Columns       : {list(df.columns)}")
print(f"\nClass distribution:")
print(df['dx'].value_counts())

# ─────────────────────────────────────────
# STEP 2: Create binary labels
# ─────────────────────────────────────────
print("\nSTEP 2: Creating binary labels...")
df['label'] = df['dx'].apply(lambda x: 1 if x == 'mel' else 0)
print(f"Malignant (mel) : {df['label'].sum()}")
print(f"Benign (others) : {(df['label'] == 0).sum()}")

# ─────────────────────────────────────────
# STEP 3: Map image paths
# ─────────────────────────────────────────
print("\nSTEP 3: Mapping image paths...")

def find_image_path(image_id):
    if pd.isna(image_id):          # NaN check
        return None
    path = os.path.join(IMAGES_DIR, str(image_id) + ".jpg")
    if os.path.exists(path):
        return path
    return None

df['image_path'] = df['image_id'].apply(find_image_path)

missing = df['image_path'].isna().sum()
print(f"Images found   : {len(df) - missing}")
print(f"Images missing : {missing}")

df = df.dropna(subset=['image_path']).reset_index(drop=True)

# ─────────────────────────────────────────
# STEP 4: Check for corrupted images
# ─────────────────────────────────────────
print("\nSTEP 4: Checking for corrupted images...")
corrupted = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying"):
    try:
        img = Image.open(row['image_path'])
        img.verify()
    except Exception:
        corrupted.append(idx)

print(f"Corrupted images found : {len(corrupted)}")
if corrupted:
    df = df.drop(index=corrupted).reset_index(drop=True)
    print(f"Removed. Remaining: {len(df)}")
else:
    print("No corrupted images found — dataset is clean!")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─────────────────────────────────────────
# STEP 5: Copy images to processed folder
# ─────────────────────────────────────────
print("\nSTEP 5: Copying images to processed folder...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying"):
    dest = os.path.join(PROCESSED_DIR, row['image_id'] + ".jpg")
    
    if row['image_path'] and os.path.exists(row['image_path']):
        shutil.copy2(row['image_path'], dest)
# ─────────────────────────────────────────
# STEP 6: Save clean CSV
# ─────────────────────────────────────────
print("\nSTEP 6: Saving clean CSV...")
clean_csv_path = os.path.join(RAW_DIR, "processed", "metadata_clean.csv")
save_cols = ['image_id', 'dx', 'label', 'dx_type', 'age', 'sex', 'localization']
df[save_cols].to_csv(clean_csv_path, index=False)
print(f"Saved: {clean_csv_path}")

# ─────────────────────────────────────────
# STEP 7: Generate charts
# ─────────────────────────────────────────
print("\nSTEP 7: Generating charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 7-class distribution
class_counts = df['dx'].value_counts()
colors = ['#e74c3c' if c == 'mel' else '#3498db' for c in class_counts.index]
axes[0].bar(class_counts.index, class_counts.values, color=colors)
axes[0].set_title("7-Class Distribution (Red = Melanoma)")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
for i, (cls, count) in enumerate(class_counts.items()):
    axes[0].text(i, count + 20, str(count), ha='center', fontsize=9)

# Binary distribution
binary_counts = df['label'].value_counts()
axes[1].pie(
    binary_counts.values,
    labels=['Benign', 'Malignant'],
    colors=['#3498db', '#e74c3c'],
    autopct='%1.1f%%',
    startangle=90
)
axes[1].set_title("Binary Distribution")

plt.tight_layout()
chart_path = os.path.join(RAW_DIR, "processed", "class_distribution.png")
plt.savefig(chart_path, dpi=150)
plt.show()
print(f"Chart saved: {chart_path}")

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("PHASE 1 COMPLETE — SUMMARY")
print("=" * 50)
print(f"Total images      : {len(df)}")
print(f"Malignant (mel)   : {df['label'].sum()}")
print(f"Benign            : {(df['label'] == 0).sum()}")
print(f"Missing removed   : {missing}")
print(f"Corrupted removed : {len(corrupted)}")
print(f"Clean CSV         : {clean_csv_path}")
print(f"Processed images  : {PROCESSED_DIR}")
print("=" * 50)
print("Phase 1 complete! Ready for Phase 2.")