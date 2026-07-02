import kagglehub
import shutil
import os

DATASET_SLUG = "xdxd003/ff-c23"

path = kagglehub.dataset_download(DATASET_SLUG)
print("Downloaded to:", path)

dest = os.path.join("ml", "data", "raw", "faceforensics")
os.makedirs(dest, exist_ok=True)

for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(dest, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Copied dataset into:", dest)