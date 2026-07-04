import sys
import os
sys.path.append(os.path.dirname(__file__))

from dataset import DeepfakeSequenceDataset

TEST_DIR = "ml/data/processed/faceforensics_faces"
LABEL_MAP = {
    "real": 0,
    "fake_deepfakes": 1,
    "fake_face2face": 1,
    "fake_faceshifter": 1,
    "fake_faceswap": 1,
    "fake_neuraltextures": 1,
}

if __name__ == "__main__":
    dataset = DeepfakeSequenceDataset(TEST_DIR, LABEL_MAP, sequence_length=16)

    print(f"Total sequences found: {len(dataset)}")

    if len(dataset) > 0:
        frames, label = dataset[0]
        print(f"Sample sequence shape: {frames.shape}")
        print(f"Sample label: {label.item()}")
    else:
        print("WARNING: No sequences found - check folder structure or sequence_length.")