This project is a pedestrian detection and counting application that uses YOLOv8 for detection, Deep SORT for tracking, and a GUI built with Tkinter. It compares detected pedestrian counts against ground truth data and computes performance metrics like MAE, MAPE, and accuracy.

Features
Detects pedestrians in video files using YOLOv8.

Tracks individuals across frames using Deep SORT.

Allows user-defined regions of interest (ROIs) for counting.

GUI for easy video selection and ROI setup.

Compares results against predefined ground truth data.

Outputs annotated video and evaluation metrics.

Requirements
Install dependencies via pip:

bash
Copy
Edit
pip install opencv-python ultralytics numpy tqdm supervision scikit-learn deep_sort_realtime
Files
proj.py: Main script that handles detection, tracking, ROI selection, evaluation, and GUI.

ground_truth.py: Provides a dictionary of ground truth pedestrian counts for evaluation.

How to Use
Run the Application

bash
Copy
Edit
python proj.py
In the GUI:

Browse and select a video file.

Choose a target folder where the output will be saved.

Enter comma-separated region names (e.g., Region1,Region2).

Select ROIs:

For each region name, draw a polygon by clicking four points in the video frame.

After drawing, press Esc or wait 30 seconds to proceed.

Detection:

The system runs detection and tracking.

Displays pedestrian counts per region.

Saves an annotated video and shows evaluation metrics in the terminal.

Output
Annotated video saved to the selected target directory.

Pedestrian counts printed in the GUI.

Evaluation metrics printed in the console:

MAE (Mean Absolute Error)

MAPE (Mean Absolute Percentage Error)

Counting Accuracy

Ground Truth Format
The ground truth data (in ground_truth.py) is structured as follows:

python
Copy
Edit
{
    frame_number: {"Region1": count1, "Region2": count2, ...},
    ...
}
Update this dictionary to match your specific video and region frame data for accurate evaluation.
