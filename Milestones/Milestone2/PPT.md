
# Big picture:
The big picture of this project
collection_app -> AI service - > game

## collection_app
Generate Realtime Spatial coordinates from camera
91 joints data - https://github.com/ouut/capstone_project/blob/master/doc/artkit_body_joint.en.md

1. For development:
csv - https://github.com/ouut/capstone_project/blob/master/doc/artkit_body_data_example.csv

2. For product:
realtime data - Live demonstration

## game
A patched switch simulator to support UDP input data
https://github.com/ouut/eden_build/blob/overlay_cpp/scripts/game_controller.ipynb
jupyter Live demonstration

## AI service
Accept coordinates and go throughout the pipeline to generate commands for games controller
rule based system and model prediction system



# Updated Technical Approach:

## Technical Evolution Roadmap

**1. End-to-End Deep Learning from Raw Video**
Initially, we started by building our own human pose estimation model from scratch. We trained the system using raw video footage, allowing the model to learn how to identify and track body movements directly from the visual data.

**2. Leveraging Native API for Skeleton Data**
To improve accuracy and efficiency, we shifted to using Apple’s native APIs. By utilizing the built-in hardware and software capabilities of the iPhone, we can directly extract precise human skeleton coordinates. We then use this high-quality skeleton data to train our own specialized motion analysis models.

**3. Embedding-Based Motion Recognition**
Our current approach focuses on advanced feature extraction. Instead of recording and labeling our own massive datasets, we utilize pre-trained third-party models to generate "embeddings" (mathematical representations) of human actions. By comparing the embeddings of in-game actions against a database of standard, expert-level movements, we can accurately evaluate performance without the need for manual data collection.


## Comparison of Technical Evolution Stages

| Stage | Approach | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **1. Raw Video Training** | Training models directly from video footage. | Complete control over data; tailored for specific use cases. | Very time-consuming; requires massive amounts of labeled data and high computing power. |
| **2. iPhone Native API** | Using built-in Apple APIs for skeletal data. | Highly accurate and stable data; avoids complex low-level visual processing. | Hardware-dependent (locked to Apple ecosystem); still requires custom logic for motion analysis. |
| **3. Embedding-Based** | Using third-party models to generate motion embeddings. | **Highest efficiency**; no need for custom data collection; fast and scalable matching. | Relies on the quality of third-party models; may miss fine-grained details specific to your needs. |


# Baseline Model / Core Prototype:
○ Demonstrate runnable code for at least a baseline ML model (e.g.,
Logistic Regression, simple CNN, basic clustering) trained on the
prepared data OR
○ Demonstrate runnable code for a core component prototype (e.g.,
a basic Flask/FastAPI endpoint structure that accepts dummy data, a
basic Dash/Streamlit dashboard layout with placeholder data/plots).
○ Discuss initial results or functionality. What worked? What didn't?


#  Progress & Planning:
○ Review progress against the plan set in Milestone 1.
○ Detailed task breakdown for the next 3 weeks (leading to Milestone
3), focusing on core model development and integration.
○ Any adjustments to team roles or workflow.
○ Revised assessment of risks.