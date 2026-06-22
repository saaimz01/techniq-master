# Batting Technique Master

A real-time computer vision tool that analyzes cricket batting stance using pose estimation. It tracks the alignment of the head, hands, and front foot through your webcam feed and scores your technique on a 0–100 scale.

## How It Works

The engine uses [MediaPipe](https://developers.google.com/mediapipe)'s Pose Landmarker to detect body keypoints in each video frame. For every frame, it extracts three key points:

- **Head** — nose landmark
- **Hands** — midpoint between left and right wrists
- **Front foot** — whichever toe (left or right) has higher visibility confidence

It then measures how far these three points deviate horizontally from their average position. Tighter horizontal alignment (head, hands, and front foot stacked closely along a vertical line) produces a higher score, reflecting a more balanced batting position.

A yellow vertical reference line marks the average horizontal position used for scoring, while blue lines connect head → hands → front foot to visualize the actual posture.

## Demo Output

While running, the app overlays on the live webcam feed:

- **Technique Score** (green if above 60, red otherwise)
- **Front foot** label (L foot / R foot)
- Keypoint markers and alignment lines

Press **Esc** to quit.

## Requirements

- Python 3.9+
- A webcam
- [MediaPipe Pose Landmarker model file](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) (`pose_landmarker_lite.task`)

### Dependencies

```bash
pip install opencv-python mediapipe numpy
```

## Setup

1. Download the `pose_landmarker_lite.task` model from the [MediaPipe model card](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models) page.
2. Update the `model_path` variable in the script to point to your local copy:

```python
model_path = "path/to/pose_landmarker_lite.task"
```

3. Run the script:

```bash
python btm02.py
```

## Scoring Logic

```
score = max(0, 100 - normalized_deviation * 400)
```

Where `normalized_deviation` is the mean absolute deviation of the head, hands, and front-foot x-coordinates from their average, normalized by frame width. Smaller deviation (better vertical alignment) yields a higher score.

## Notes & Limitations

- Designed for a single batter in frame; only the first detected pose is used.
- Detection requires a visibility confidence above `0.2` for the nose, both wrists, and both toes — frames below this threshold are skipped.
- Currently uses `pose_landmarker_lite`; swapping in the `full` or `heavy` model may improve accuracy at the cost of speed.
- This is a basic geometric heuristic, not a substitute for coaching feedback — it's intended as a lightweight, real-time visual aid.

## Possible Improvements

- Support video file input in addition to live webcam feed
- Track technique score over time / across shots
- Add configurable thresholds for visibility and scoring sensitivity
- Multi-person handling for analyzing footage with multiple batters
