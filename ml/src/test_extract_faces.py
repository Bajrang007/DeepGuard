"""
Quick test: runs face extraction on just 3 real + 3 fake videos,
to verify the pipeline works correctly before running on the full dataset.
"""

import os
import shutil
import sys

sys.path.append(os.path.dirname(__file__))
from extract_faces import process_folder

REAL_VIDEOS_DIR = "ml/data/raw/faceforensics/FaceForensics++_C23/original"
FAKE_VIDEOS_DIR = "ml/data/raw/faceforensics/FaceForensics++_C23/Deepfakes"
TEST_OUTPUT_DIR = "ml/data/processed/test_faces"

# Create a temporary folder with just 3 sample videos copied in
def make_sample_folder(source_dir, dest_dir, count=3):
    os.makedirs(dest_dir, exist_ok=True)
    files = sorted(os.listdir(source_dir))[:count]
    for f in files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(dest_dir, f))
    return dest_dir

if __name__ == "__main__":
    sample_real = make_sample_folder(REAL_VIDEOS_DIR, "ml/data/processed/sample_real", count=3)
    sample_fake = make_sample_folder(FAKE_VIDEOS_DIR, "ml/data/processed/sample_fake", count=3)

    process_folder(sample_real, TEST_OUTPUT_DIR, "real")
    process_folder(sample_fake, TEST_OUTPUT_DIR, "fake")

    print("\nDone. Check ml/data/processed/test_faces/ for extracted face images.")