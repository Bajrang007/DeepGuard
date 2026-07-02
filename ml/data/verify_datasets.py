import os, sys

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")

def check_asvspoof():
    base = os.path.join(RAW_DIR, "asvspoof2019", "LA")
    if not os.path.isdir(base):
        print("[FAIL] ASVspoof not found at", base)
        return False
    protocol_dir = os.path.join(base, "ASVspoof2019_LA_cm_protocols")
    train_audio_dir = os.path.join(base, "ASVspoof2019_LA_train", "flac")
    ok = os.path.isdir(protocol_dir) and os.path.isdir(train_audio_dir)
    if ok:
        import soundfile as sf
        files = [f for f in os.listdir(train_audio_dir) if f.endswith(".flac")]
        if files:
            data, sr = sf.read(os.path.join(train_audio_dir, files[0]))
            print(f"[PASS] Loaded {files[0]}: {len(data)} samples @ {sr}Hz")
        else:
            print("[FAIL] No .flac files found")
            ok = False
    else:
        print("[FAIL] Missing protocol or audio directory")
    return ok

def check_faceforensics():
    base = os.path.join(RAW_DIR, "faceforensics")
    if not os.path.isdir(base) or not os.listdir(base):
        print("[PENDING] FaceForensics++ not yet approved/downloaded")
        return None
    print("[PASS] FaceForensics++ present")
    return True

if __name__ == "__main__":
    asv_ok = check_asvspoof()
    check_faceforensics()
    sys.exit(0 if asv_ok else 1)