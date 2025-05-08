# 🧍‍♂️ Pedestrian Detection and Counting System

This project provides a GUI-based tool to detect and count pedestrians in video footage using YOLOv8 and Deep SORT. It allows users to define Regions of Interest (ROIs) manually, track individuals through frames, and evaluate the predicted counts against ground truth data.

---

## 📂 Project Structure

pedestrian-detection/
├── ground_truth.py # Ground truth pedestrian counts per frame
├── proj.py # Main GUI application and detection pipeline
├── requirements.txt # Python dependencies
├── README.md # This documentation
├── .gitignore # Git ignored files



---

## 🔧 Features

- GUI interface for video selection and ROI setup.
- Pedestrian detection using [YOLOv8](https://github.com/ultralytics/ultralytics).
- Object tracking via [Deep SORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch).
- Draw and define custom regions to monitor pedestrian traffic.
- Real-time count overlay on video.
- Annotated video output.
- Performance evaluation with:
  - Mean Absolute Error (MAE)
  - Mean Absolute Percentage Error (MAPE)
  - Accuracy %

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/pedestrian-detection.git
cd pedestrian-detection


Install dependencies:

pip install -r requirements.txt

## Usage
Run the application:

bash
Copy
Edit
python proj.py
In the GUI:

Click “Browse” to select a video file.

Click “Browse” to choose a target folder where results will be saved.

Enter region names (comma-separated, e.g., Region1,Region2) and click “Start Detection”.

Draw ROIs:

For each region, select 4 points by clicking in the video frame.

Press Esc or wait 30 seconds to move to the next.

Results:

Annotated video saved in your target folder.

Counts are shown in the GUI.

Evaluation metrics are printed in the terminal.

📊 Ground Truth
Update ground_truth.py with frame-based expected counts for evaluation:

python
Copy
Edit
def get_ground_truth():
    return {
        0: {"Region1": 5, "Region2": 3},
        50: {"Region1": 8, "Region2": 6},
        100: {"Region1": 12, "Region2": 9},
        ...
    }
Only frames with entries in this dictionary are used for metric calculations.

🧪 Evaluation Metrics
Displayed in the terminal after video processing:

MAE: Mean difference between predicted and actual counts.

MAPE: Percentage error of the predictions.

Accuracy: 100 - MAPE, capped at 0 if MAPE > 100%.

