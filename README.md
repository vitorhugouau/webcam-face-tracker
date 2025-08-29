# Webcam Face Tracker

Real-time face detection and tracking via webcam—with smart event-based video recording.

---

##  Features

- **Live face detection and tracking** using MTCNN (via `align.detect_face`).
- **Automatic video recording** when a face is detected.
- **Graceful closing** of video after 5 seconds without re-detection.
- **MP4 output** with timestamped, collision-resistant filenames.
- Solid resource cleanup: all `VideoWriter`, webcam, and GUI windows are properly managed.
- Configurable detection interval, recording duration, thresholds, and more.

---

##  Getting Started

###  Prerequisites

- Python 3.11+ installed
- Webcam access enabled
- GPU recommended but not mandatory (for TensorFlow/numba performance)

###  Installation

```bash
git clone https://github.com/vitorhugouau/webcam-face-tracker.git
cd webcam-face-tracker
pip install -r requirements.txt

