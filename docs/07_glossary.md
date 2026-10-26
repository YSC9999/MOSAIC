# 7. Glossary: Plain-English Definitions of Technical Terms

This glossary explains every concept and acronym used in the codebase and plan, written specifically for someone new to computer vision and machine learning.

---

## Video & Computer Vision Basics

### Frame
A single still photograph within a video. A video is just a rapid sequence of frames played one after another.

### FPS (Frames Per Second)
The rate at which frames are displayed. At 30 FPS, thirty still photos pass your eyes every second. In MOSAIC-FL, each frame's timestamp is calculated as `frame_index / FPS` so timestamps are consistent and exact across runs.

### Resolution (Width × Height)
The number of pixels (dots) in each frame. For example, `160 × 120` means 160 pixels wide and 120 pixels high. Common standards include 1080p (1920×1080), 720p (1280×720), and 240p (320×240).

### BGR vs. RGB
Color images are made of three color channels: Red, Green, and Blue. Most software orders them RGB, but OpenCV historically stores them as **BGR** (Blue, Green, Red). A frame array has dimensions `(Height, Width, 3)`.

### Bounding Box (Box)
A rectangle drawn around an object of interest (like a person). We represent it with four pixel coordinates `[x1, y1, x2, y2]`, where `(x1, y1)` is the top-left corner and `(x2, y2)` is the bottom-right corner.

### Centroid
The exact geometric center point `(cx, cy)` of a bounding box.

### Occlusion
When an object is partially or completely hidden from camera view by another object or person (e.g., someone walking behind a wall or standing behind another person in a crowd).

### Silhouette / Mask
A pixel-accurate outline of a person's shape, as opposed to a simple rectangular box. Silhouettes show limb movements (arms waving, kicking) even when clothing or shadows obscure detailed body textures.

---

## Detection, Tracking & Motion

### Object Detection
A computer vision model that looks at a single still frame and outputs bounding boxes plus confidence scores for objects it recognizes (e.g. "person: 92% confident").

### Multi-Object Tracking (MOT)
Connecting detections across consecutive frames to follow individuals over time. Detection only knows *where* a person is in frame 10; Tracking knows that person in frame 10 is the *same person* seen in frame 9, assigning them a persistent `track_id`.

### Kalman Filter
A mathematical algorithm that estimates the true position and velocity of a moving object and predicts where it will be in the next frame. If a person is temporarily hidden (occluded) or a frame is skipped, the Kalman filter predicts their position.

### Hungarian Algorithm
A matching algorithm that solves the assignment problem: pairing the set of predicted tracks from the previous frame with the set of new detections in the current frame to decide who is who.

### Motion Gating
A technique to save computational effort. If frame 2 is 95% identical to frame 1, running a heavy neural network on frame 2 is redundant. The motion gate detects that nothing moved and skips running the detector on that frame.

### Track Token
The core standardized data format connecting tracking to downstream analysis. For every person in every processed frame, a Track Token packages their ID, timestamp, box, centroid, velocity, acceleration, and visibility status.

---

## Machine Learning & Deep Learning Basics

### Machine Learning (ML)
Writing algorithms that learn patterns from data rather than following hard-coded rules.

### Deep Learning (DL)
A subfield of ML using multi-layered artificial neural networks (inspired loosely by brain networks) capable of understanding images, audio, and text.

### Pretrained Model
A neural network that has already been trained on a massive general dataset (such as millions of images from COCO or ImageNet) by researchers with supercomputers. We can download and use its learned visual representations directly.

### Frozen Weights
Using a pretrained model in "read-only" mode. The model processes images and extracts features, but its internal parameters are not modified or trained further during our pipeline's training.

### Temporal Model (e.g., GRU)
A neural network designed for sequences over time. While an image detector only sees one instant, a temporal model takes a history of observations across several seconds to understand what is happening (e.g., recognizing that rapid closing distance followed by abrupt deceleration represents a physical strike).

### Weak Supervision & Video-Level Labels
In RWF-2000, clips only have one broad label: "violent" or "non-violent". No one annotated individual seconds, drawn boxes, or indicated who initiated the fight. Learning fine-grained patterns from coarse labels is called *weak supervision*.

---

## Software Engineering & Testing Terminology

### Data Contract
A formal specification in code describing what valid data must look like. If an upstream function creates an impossible value (like a negative box area or a confidence of 1.5), the contract raises an error immediately at the boundary.

### Determinism & Random Seed
Computers do not generate truly random numbers; they use formulas starting from an initial number called a *seed*. Setting the seed to a fixed value (e.g., 1337) guarantees that pseudo-random operations (like data shuffling or synthetic texture generation) generate the exact same numbers every time the code runs.

### Ground Truth (GT)
The objective, verified "answer key". For synthetic videos, we program the exact coordinates where rectangles are drawn, giving us 100% accurate ground truth for testing.

### Fixture
Sample test data or objects set up in advance so tests can run against known, repeatable inputs without needing live camera feeds or external downloads.

### Unit Test
A test that verifies a single small piece of code (like a single function or class method) in isolation.

### Integration Test
A test that verifies that multiple modules work together smoothly across their connections.

### Smoke Test
A fast, end-to-end check to verify that the basic pipeline runs without crashing, producing the expected output files and schemas.
