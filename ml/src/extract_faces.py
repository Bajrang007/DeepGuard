"""
Extracts face crops from ALL FaceForensics++ categories (original + all
4 manipulation types), using OpenCV's DNN face detector.
"""

import os
import cv2
from tqdm import tqdm

FRAME_SAMPLE_RATE = 5
CROP_SIZE = 224
PROTOTXT_PATH = "ml/models/face_detector/deploy.prototxt"
MODEL_PATH = "ml/models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
CONFIDENCE_THRESHOLD = 0.5

BASE_DIR = "ml/data/raw/faceforensics/FaceForensics++_C23"
OUTPUT_DIR = "ml/data/processed/faceforensics_faces"

CATEGORIES = {
    "original": "real",
    "Deepfakes": "fake_deepfakes",
    "Face2Face": "fake_face2face",
    "FaceShifter": "fake_faceshifter",
    "FaceSwap": "fake_faceswap",
    "NeuralTextures": "fake_neuraltextures",
}


def create_detector():
    return cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)


def extract_faces_from_video(video_path, output_dir, net):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SAMPLE_RATE == 0:
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                          (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            best_confidence = 0
            best_box = None

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > CONFIDENCE_THRESHOLD and confidence > best_confidence:
                    box = detections[0, 0, i, 3:7] * [w, h, w, h]
                    best_box = box.astype("int")
                    best_confidence = confidence

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (CROP_SIZE, CROP_SIZE))
                    out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                    cv2.imwrite(out_path, face_crop)
                    saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count


def process_folder(input_folder, output_folder, label):
    os.makedirs(output_folder, exist_ok=True)
    net = create_detector()

    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    for video_file in tqdm(video_files, desc=f"Processing {label}"):
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(input_folder, video_file)
        out_dir = os.path.join(output_folder, label, video_id)

        count = extract_faces_from_video(video_path, out_dir, net)
        if count == 0:
            print(f"WARNING: No faces detected in {video_file}")


if __name__ == "__main__":
    for folder_name, label in CATEGORIES.items():
        input_dir = os.path.join(BASE_DIR, folder_name)
        print(f"\n=== Starting category: {folder_name} -> label: {label} ===")
        process_folder(input_dir, OUTPUT_DIR, label)

    print("\nAll categories processed.")