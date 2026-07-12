\# Lip-Sync Fusion Module



Uses the pretrained SyncNet model (joonson/syncnet\_python) to score

audio-visual synchronization as a third detection signal, alongside

the video (Phase 2) and audio (Phase 3) models.



\## Setup (Colab/deployment environment)

1\. Clone: https://github.com/joonson/syncnet\_python.git

2\. Download pretrained model: sh download\_model.sh (inside that repo)

3\. Copy lipsync\_score.py (this file) into that repo's root directory

4\. Run: python lipsync\_score.py (or import get\_sync\_score from it)



\## Note

This module assumes input video is already face-cropped/tracked

(a talking-head style clip), consistent with SyncNet's expected input

format. Full-frame face detection/cropping happens upstream (Phase 6

integration) before calling this module.



\## Validation

See evaluation\_results.txt for validation approach and results

(controlled desync test - synced confidence 10.081 vs desynced 8.830).

Full validation against real audio-visual deepfakes deferred to a

post-Phase-9 follow-up task.

