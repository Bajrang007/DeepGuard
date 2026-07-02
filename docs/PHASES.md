# DeepGuard Phase Plan

0. Foundations - datasets, tracking, repo structure
1. Core Calling Infra - WebRTC, Socket.io, coturn
2. Video Detection Model - EfficientNet-B0 + GRU
3. Audio Detection Model - RawNet2 on ASVspoof
4. Lip-Sync Fusion - SyncNet
5. On-Device Inference - ONNX Runtime Web
6. Server-Side Confirmation - FastAPI + TorchServe
7. Decision Engine - MLP fusion, calibration
8. Frontend/UX - React alert UI
9. Feedback and Data Loop - PostgreSQL
10. Evaluation Harness - AUC, EER, cross-dataset test
11. Deployment - Docker Compose, Vercel, Render
12. Documentation - diagrams, model cards, demo video
