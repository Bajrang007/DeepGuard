"""
Clean wrapper around SyncNet for getting a lip-sync confidence score
from a video clip. Assumes input video is already face-cropped/tracked
(e.g., output of run_pipeline.py, or a pre-cropped talking-head clip).
"""

import os
import types
from SyncNetInstance import SyncNetInstance

_model = None


def load_model(model_path="data/syncnet_v2.model"):
    """Loads the SyncNet model once, reused across calls."""
    global _model
    if _model is None:
        _model = SyncNetInstance()
        _model.loadParameters(model_path)
        print(f"SyncNet model loaded from {model_path}")
    return _model


def get_sync_score(video_path, tmp_dir="/content/temp_sync", batch_size=20, vshift=15):
    """
    Returns a sync confidence score for a given (already face-cropped) video clip.
    Higher score = better audio-visual sync. Lower score = likely desync/mismatch.
    """
    model = load_model()

    opt = types.SimpleNamespace(
        tmp_dir=tmp_dir,
        reference=os.path.splitext(os.path.basename(video_path))[0],
        batch_size=batch_size,
        vshift=vshift,
    )

    offset, conf, dists = model.evaluate(opt, video_path)

    return {
        "confidence": float(conf),
        "av_offset": int(offset),
    }


if __name__ == "__main__":
    result = get_sync_score("data/example.avi")
    print("Sync result:", result)
