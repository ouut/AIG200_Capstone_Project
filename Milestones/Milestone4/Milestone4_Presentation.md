# MotionPlay: A Camera-Based Body-Motion Game Controller

## Milestone 4 Presentation — July 24, 2026

**Team:** Kunal Pandey, Lei Li, Chao Chen, Chenghua Jiang

**AIG200 Capstone Project — Spring/Summer 2026**

---

<!-- SLIDE 1 -->

## Slide 1: What is MotionPlay?

**Play games with your body — no controller needed.**

- An iPhone camera captures your body movements using ARKit
- A machine learning model recognizes your gestures in real time
- Your movements become game controller commands
- Works with Nintendo Switch emulators (Eden/Yuzu)

**The big idea:** Instead of pressing buttons, you jump, dash, and move your body to control the game character.

---

<!-- SLIDE 2 -->

## Slide 2: The Three System Components

Our system is built as three independent pieces that work together:

| Component | What It Does | Technology |
|---|---|---|
| **iOS Capture App** | Tracks 91 body joints at 60 FPS, streams data over network | Swift, ARKit 4 |
| **ML Training Pipeline** | Trains a gesture recognition model on dance + custom data | PyTorch, ST-GCN |
| **Real-Time Server** | Runs live inference, sends game controller commands | Python, asyncio, Docker |

Each component is in its own repository and can be developed and tested independently.

---

<!-- SLIDE 3 -->

## Slide 3: How the Data Flows

```
Player Moves
     ↓
iPhone Camera + ARKit  (91 joints, 60 FPS)
     ↓
Network Stream  (UDP / TCP / WebSocket, 2652-byte frames)
     ↓
Preprocessing  (12 joints, centered, normalized, 300 frames)
     ↓
ST-GCN Model  (256-dimensional embedding)
     ↓
Cosine Similarity → Template Bank Match
     ↓
DSU Protocol → Nintendo Switch Emulator  (UDP port 26760)
     ↓
Game Character Moves!
```

**Total latency: under 100 milliseconds from movement to game action.**

---

<!-- SLIDE 4 -->

## Slide 4: The iOS Capture App

**What it does:**
- Uses the iPhone's TrueDepth camera to track a full body skeleton
- Shows a 3D robot character that mirrors your movements in real time
- Records CSV files with joint positions and rotations
- Records video (HEVC format)

**How it sends data:**
- **UDP:** Fastest, no reconnection needed
- **TCP:** Reliable, with automatic reconnection
- **WebSocket:** Works through firewalls, supports browsers

**Binary protocol:** 2652 bytes per frame, containing 91 joints (position + rotation) and camera pose.

**Requires:** iPhone with A12 chip or newer, iOS 15+

---

<!-- SLIDE 5 -->

## Slide 5: The ML Training Pipeline (3 Stages)

### Stage 1: Pretraining
- Train ST-GCN on **1,408 AIST++ dance clips** (10 dance styles)
- Model learns to understand human body movement patterns
- Achieves **~93% accuracy** on dance classification

### Stage 2: Template Bank
- Collect custom gesture recordings from 3 subjects
- Compute a 256-dimension embedding for each recording
- Average embeddings per gesture → **gesture template bank**

### Stage 3: Evaluation
- Leave-one-subject-out cross-validation
- **61% accuracy** across all 6 gestures
- **73% accuracy** for the 4 best gestures

---

<!-- SLIDE 6 -->

## Slide 6: What is ST-GCN?

**Spatial-Temporal Graph Convolutional Network**

- A neural network designed for skeleton-based action recognition
- **Spatial:** Learns relationships between connected joints (e.g., elbow connects to wrist and shoulder)
- **Temporal:** Learns how those relationships change over time
- Treats the human skeleton as a graph with 12 nodes (joints) and edges (bones)

**Why we chose it:**
- Works directly on joint coordinates — no need for video frames
- Handles different body sizes through normalization
- Proven on public benchmarks (NTU RGB+D, Kinetics-Skeleton)

**Transfer learning approach:** Pretrain on a large public dataset (AIST++), then fine-tune on our small custom dataset.

---

<!-- SLIDE 7 -->

## Slide 7: The Real-Time Server

**How it works:**

1. Receives ARKit body frames over the network
2. Preprocesses each frame (maps joints, normalizes, resamples)
3. Runs **two recognition methods in parallel:**
   - **Rule-based tick functions:** Analyze joint angles and accelerations frame by frame
   - **ST-GCN embedding matching:** Compare current movement to gesture templates
4. Gestures compete within groups using **softmax scoring**
5. Winning gesture triggers an **action sequence** (e.g., "press A for 4 frames, hold stick left")
6. Sends controller state via **DSU protocol** (60 Hz) to the emulator

**Bonus features:**
- Web dashboard with live 3D skeleton and gesture scores (port 8080)
- Virtual gamepad page for keyboard fallback control
- Fully Dockerized — deploy with one command

---

<!-- SLIDE 8 -->

## Slide 8: Live Demonstration

### What we will show:

1. **iPhone app running** — 3D robot character mirroring the player's movements
2. **Web dashboard** — Real-time 3D skeleton, gesture scores, and button states
3. **Gameplay** — Playing Super Mario 3D World using body gestures

### Gesture mapping:

| Body Movement | Game Action | Button |
|---|---|---|
| Jump up | Mario jumps | A |
| Quick crouch / squat | Mario dashes | B |
| Lean / step left | Move left | Left Stick ← |
| Lean / step right | Move right | Left Stick → |

---

<!-- SLIDE 9 -->

## Slide 9: Testing & What We Found

### What we tested:

| Test Type | Details |
|---|---|
| **Unit Tests** | 223 DSU protocol tests, system component tests |
| **Model Evaluation** | Cross-user accuracy: 61% (6 gestures), 73% (4 gestures) |
| **Integration Tests** | CSV replay mode for repeatable testing without iPhone |
| **End-to-End** | Full pipeline tested with actual gameplay |

### Key findings:

- **Jump and Dash work well** — detected by wrist acceleration and hip drop
- **Left and Right work well** — detected by arm angle and body lean
- **Forward and Backward are difficult** — ARKit reports positions relative to the hip, which cancels out forward movement
- **Jump height is invisible to ARKit** — less than 2 cm of vertical hip movement detected

### Bugs we fixed:
- Video recording crash (pixel buffer lifetime issue)
- WebSocket unmasked frames
- ST-GCN input shape mismatch
- Gesture bank normalization

---

<!-- SLIDE 10 -->

## Slide 10: Deployment & Final Plan

### Deployment

| Component | How It's Deployed |
|---|---|
| iOS App | Direct install via `ios-deploy`, runs on device |
| Server | Docker container (Docker Hub: `chet2026/motionplay`) |
| Model | Weights bundled in Docker image (~3.6 MB) |

**One-command server deployment:**
```bash
docker compose -f docker-compose.hub.yml up -d
```

### What's left (final 2 weeks):

1. **Final report** — 5-page document with full system description
2. **Final presentation** — Polish slides and practice delivery
3. **Accuracy improvement** — Re-record Forward/Jump gestures with upper-body protocol
4. **Demo video** — Record end-to-end gameplay for the submission
5. **Documentation polish** — Review all README files and doc pages

### Risks:
- Jump gesture may remain unreliable → document as a known limitation
- Time is tight → prioritize report and presentation over optional features

---

## Thank You! Questions?

**GitHub:** [github.com/ouut/AIG200_Capstone_Project](https://github.com/ouut/AIG200_Capstone_Project)

**Docker Hub:** `chet2026/motionplay:latest`
