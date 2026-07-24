# MotionPlay: A Camera-Based Body-Motion Game Controller

## Milestone 4 — Near-Complete System & Deployment Report

**Team:** Kunal Pandey (Data & Capture), Lei Li (ML), Chao Chen (Systems), Chenghua Jiang (Documentation & QA)

**Course:** AIG200 Capstone Project — Spring/Summer 2026

**Date:** July 24, 2026

---

## 1. Project Overview

MotionPlay is an end-to-end system that lets a player control a video game using body movements instead of a traditional controller. An iPhone camera captures the player's body pose using Apple's ARKit. The pose data is streamed to a computer, where a machine learning model recognizes the player's gestures. Those gestures are then translated into game controller commands and sent to a Nintendo Switch emulator.

The project is divided into three main parts, each in its own repository:

| Component | Repository | Purpose |
|---|---|---|
| iOS Capture App | [`capstone_app`](https://github.com/ouut/capstone_app) | Captures body pose and streams data over the network |
| ML Training Pipeline | [`capstone_training`](https://github.com/ouut/capstone_training) | Trains a gesture recognition model and builds a gesture template bank |
| Real-Time Server | [`capstone_server`](https://github.com/ouut/capstone_server) | Runs live inference and sends game controller commands |
| Emulator Config | [`capstone_eden`](https://github.com/ouut/capstone_eden) | Nintendo Switch emulator setup and configuration |

### Target Gestures

The system recognizes six body gestures: **Jump, Move Left, Move Right, Crouch/Dash, Move Forward, and Move Backward**. Four of the six gestures (Dash/Crouch, Move Left, Move Right, Move Backward) work reliably through the ST-GCN model. Two gestures (Move Forward and Jump) are difficult for the ML model to detect because ARKit reports body joint positions relative to the hip, which cancels out forward movement and vertical jumping motion. However, Jump is recovered in the real-time server through rule-based detection that looks at wrist acceleration instead of hip height.

---

## 2. End-to-End System Architecture

The complete data flow from the player's body to the game character is shown below:

```
[Player Moves] → [iPhone Camera + ARKit]
       ↓
  91 joints tracked at 60 FPS
       ↓
[Network Streaming: UDP / TCP / WebSocket]
       ↓ (2652-byte binary frames)
[Real-Time Server (Python + asyncio)]
       ↓
  Preprocessing Pipeline:
  - Map 91 ARKit joints → 12 COCO joints
  - Align orientation to first frame
  - Center at hip, normalize by torso length
  - Resample to 300 frames
       ↓
  Gesture Recognition (two methods):
  A) Rule-based tick functions (per-frame joint angle/acceleration analysis)
  B) ST-GCN embedding + cosine similarity to gesture template bank
       ↓
  Grouped softmax → winning gesture
       ↓
[DSU Protocol Server (UDP port 26760)]
       ↓
[Nintendo Switch Emulator (Eden/Yuzu)]
       ↓
[Game Character Performs the Action]
```

### 2.1 iOS Capture App (`capstone_app`)

- Built with Swift and ARKit 4 (`ARBodyTrackingConfiguration`)
- Tracks 91 body joints at 60 frames per second
- Renders a 3D robot character on screen that mirrors the player's movements
- Records CSV files (joint positions and rotations over time)
- Records video (HEVC/H.265 encoding)
- Streams skeletal data over UDP, TCP, or WebSocket using a custom 2652-byte binary protocol
- Supports UDP video streaming (MJPEG) for remote monitoring
- Includes a settings interface for configuring all network transports
- Requires an iPhone with an A12 chip or newer (iOS 15+)

### 2.2 ML Training Pipeline (`capstone_training`)

- **Stage 1 — Pretraining:** An ST-GCN (Spatial-Temporal Graph Convolutional Network) is trained on the AIST++ dance dataset (1,408 clips, 10 dance styles) to classify dance styles. This teaches the model to understand human body movement.
- **Stage 2 — Template Bank:** The pretrained model is used to compute a 256-dimensional embedding for each custom gesture recording. Embeddings for the same gesture are averaged to create a template. The result is stored in `gesture_bank.json`.
- **Stage 3 — Evaluation:** Cross-user evaluation (leave-one-subject-out) measures recognition accuracy. Overall accuracy on 6 gestures is 61%. With the 4 most reliable gestures, accuracy reaches 73%.

### 2.3 Real-Time Server (`capstone_server`)

- Written in Python using asyncio for concurrent network I/O
- Receives ARKit body frames from the iOS app via UDP, TCP, or WebSocket
- Runs the ST-GCN model for embedding-based gesture recognition
- Also uses rule-based tick functions (Python files that analyze joint positions frame by frame)
- Gestures compete within groups using a softmax function
- The winning gesture triggers an action sequence (a string like `A4_B16_StickL:1:-1:10`)
- A DSU (Cemuhook) protocol server sends controller state to the emulator at 60 Hz on UDP port 26760
- A web dashboard (port 8080) shows a real-time 3D skeleton, gesture scores, and button states
- A virtual gamepad page allows keyboard control as a fallback
- Fully Dockerized for easy deployment

---

## 3. Deployment Status

### 3.1 iOS App

The iOS capture app is installed directly on an iPhone via `ios-deploy`. A pre-built IPA file is available at `capstone_app/build/ipa/collection_app.ipa`. The app runs locally on the device and streams data over the local network. It does not require cloud deployment.

### 3.2 Real-Time Server (Docker)

The server is fully Dockerized and can be deployed on any machine with Docker installed:

- **Dockerfile:** Based on `python:3.10-slim`, installs CPU-only PyTorch and dependencies
- **Docker Compose:** Two compose files are provided — one builds from source, the other pulls a pre-built image from Docker Hub (`chet2026/motionplay:latest`)
- **Ports exposed:** 8080 (dashboard), 26760 (DSU controller), 8765 (ARKit data input)

Deployment command:
```bash
docker compose -f docker-compose.hub.yml up -d
```

### 3.3 Deployment Challenges

| Challenge | Solution |
|---|---|
| ARKit coordinate system is hip-relative | Preprocessing re-centers and normalizes all skeletons to a common frame |
| Different users have different body sizes | Torso-length normalization makes the system body-size independent |
| Network latency between phone and server | UDP transport chosen for minimum latency; TCP available for reliability |
| PyTorch GPU dependencies are large | CPU-only PyTorch used in Docker, reducing image size to ~275 MB |
| Cross-user gesture variability | Template bank built from multiple subjects; leave-one-subject-out evaluation used |

---

## 4. Testing & Validation

### 4.1 Unit Testing

- **DSU Protocol Tests:** 223 unit tests in `test_udp_controller.py` verify the DSU server's message handling, protocol compliance, and frame buffer logic.
- **System Component Tests:** `test_realtime.py` covers the core frame processing pipeline.
- **Python Parser Tests:** The `arkit_parser.py` library in `capstone_app/doc/` is tested with real CSV recordings, verifying correct parsing of all 91 joints and the binary protocol format.

### 4.2 Model Evaluation

- **Pretraining Accuracy:** The ST-GCN achieved ~93% training accuracy on the AIST++ 10-class dance style classification task.
- **Gesture Recognition Accuracy:** On the custom gesture dataset with leave-one-subject-out cross-validation, overall accuracy is 61% across all 6 gestures and 73% across the 4 best-performing gestures (Dash/Crouch, Left, Right, Back).
- **Confusion Matrix Analysis:** Move Forward and Move Backward are frequently confused with the idle/normal pose because the hip-relative coordinate system cancels out forward motion.

### 4.3 Integration Testing

- **CSV Replay Mode:** Both `capstone_training` and `capstone_server` support CSV replay, allowing testing without a live iPhone.
- **End-to-End Testing:** The full pipeline (iPhone → Server → Emulator) has been tested with actual gameplay in Super Mario 3D World.
- **Mock Sender:** `capstone_app/doc/example_mock_sender.py` replays recorded CSV files as network data, enabling repeatable integration tests.

### 4.4 Diagnostic Tools

Three diagnostic scripts were developed to understand gesture confusion:
- `diagnose_vertical.py` — Revealed that ARKit's hip-relative coordinates show less than 2 cm of vertical hip movement during a jump, making Jump nearly invisible to the model.
- `diagnose_lean.py` — Showed that forward body lean (~13°) is indistinguishable from normal standing posture.
- `diagnose_orientation.py` — Visualized raw body orientation to help design the alignment step in preprocessing.

### 4.5 Bug Fixes

During testing, the following issues were found and fixed:
- **Camera pixel buffer lifetime:** Video recording crashed when ARKit released pixel buffers before `AVAssetWriter` finished with them. Fixed by copying pixel buffers before enqueuing.
- **WebSocket framing:** The custom WebSocket implementation initially sent unmasked frames. Fixed by adding proper RFC 6455 masking.
- **ST-GCN input shape:** The model expected (N, C, T, V) but preprocessing produced (T, V, C). Fixed by adding a transpose step.
- **Gesture bank normalization:** Embeddings in the template bank were not L2-normalized, causing inconsistent cosine similarity scores. Fixed by normalizing before saving.

---

## 5. Documentation & Final Steps

### 5.1 Existing Documentation

| Document | Location | Language |
|---|---|---|
| iOS App README | `capstone_app/README.md` | English |
| Training Pipeline README | `capstone_training/README.md` | English |
| Training Pipeline README | `capstone_training/README.zh.md` | Chinese |
| Server README | `capstone_server/README.md` | English |
| Server README | `capstone_server/README.zh.md` | Chinese |
| Joint Index Reference | `capstone_app/doc/joint_index.md` | English |
| Joint Visual Guide | `capstone_app/doc/joint.en.md` | English |
| Joint Visual Guide | `capstone_app/doc/joint.zh.md` | Chinese |
| Network Protocol Spec | `capstone_app/doc/joint_index.md` | English |
| ARKit Parser Library | `capstone_app/doc/arkit_parser.py` | Code + comments |
| ST-GCN Training Doc | `capstone_server/doc/` | English & Chinese |

### 5.2 Remaining Work (Final 2 Weeks)

| Task | Priority | Owner | Estimated Effort |
|---|---|---|---|
| Finish Milestone 4 report and presentation slides | High | All | 2 days |
| Re-record Jump and Forward gestures with upper-body protocol | Medium | Kunal | 1 day |
| Improve gesture recognition accuracy above 80% | Medium | Lei | 2 days |
| Polish web dashboard UI (add game state display) | Low | Chao | 1 day |
| Write final project report (5+ pages) | High | All | 3 days |
| Prepare final presentation (10 slides, 10 minutes) | High | All | 2 days |
| Record demo video showing end-to-end gameplay | Medium | Chenghua | 1 day |
| Final Docker image testing and Docker Hub push | Low | Chao | 0.5 day |

### 5.3 Remaining Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Jump gesture remains unreliable | Medium | Medium | Switch to upper-body gesture protocol for Jump |
| Forward/Backward gestures cannot be distinguished | High | Low | Accept limitation; document clearly in final report |
| Cross-user accuracy drops with new testers | Low | Medium | Expand template bank with more subjects |
| Presentation time limit (10 minutes) | Medium | Medium | Practice and time the presentation; prepare backup slides |

---

## 6. Conclusion

MotionPlay is a functional, end-to-end body-motion game controller system. The three components — iOS capture app, ML training pipeline, and real-time server — are integrated and working together. The system can capture body pose from an iPhone, stream it over the network, recognize gestures using a pretrained ST-GCN model, and send game controller commands to a Nintendo Switch emulator in real time.

The project demonstrates the complete ML lifecycle: data collection, preprocessing, model training, evaluation, deployment (Docker), and real-time inference. While two of the six gestures remain challenging due to the limitations of ARKit's hip-relative coordinate system, the four core gameplay gestures (Jump, Dash, Left, Right) work reliably.

The remaining two weeks will focus on final documentation, presentation preparation, and optional accuracy improvements. The team is on track to deliver a polished, well-documented final submission.
