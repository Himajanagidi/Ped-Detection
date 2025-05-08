import cv2
from ultralytics import YOLO
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
import supervision as sv
import time
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from ground_truth import get_ground_truth
from deep_sort_realtime.deepsort_tracker import DeepSort

def evaluate_predictions(true_counts, predicted_counts):
    mae = round(mean_absolute_error(true_counts, predicted_counts), 2)
    mape = round(mean_absolute_percentage_error(true_counts, predicted_counts) * 100, 2)
    accuracy = round(100 - mape if mape < 100 else 0, 2)
    return mae, mape, accuracy

def extract_roi_from_video(video_path, regions):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(region_name, img)

    ROIs = []
    print(f'Extracting ROIs from {video_path} with {len(regions)} regions')

    for region_name in regions:
        video_info = sv.VideoInfo.from_video_path(video_path)
        generator = sv.get_video_frames_generator(video_path)
        frame = next(iter(generator))

        img = frame.copy()
        cv2.namedWindow(region_name)
        points = []
        cv2.setMouseCallback(region_name, mouse_callback, points)

        start_time = time.time()
        timeout = 30

        while True:
            cv2.imshow(region_name, img)
            key = cv2.waitKey(1)
            if key == 27 or len(points) == 4 or (time.time() - start_time > timeout):
                break

        cv2.destroyAllWindows()
        if len(points) < 4:
            print(f"No ROI selected for {region_name}. Skipping...")
            continue

        roi_x = min(pt[0] for pt in points)
        roi_y = min(pt[1] for pt in points)
        roi_width = max(pt[0] for pt in points) - roi_x
        roi_height = max(pt[1] for pt in points) - roi_y

        rectangle_range = [[roi_x, roi_x + roi_width], [roi_y, roi_y + roi_height]]

        ROIs.append({"name": region_name, "polygon": points, "range": rectangle_range})

    return ROIs

def detect_pedestrians(video_path, target_dir, regions):
    model = YOLO('yolov8x.pt')
    tracker = DeepSort(max_age=30)

    video = cv2.VideoCapture(video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing video: {video_path} at {fps} FPS, {width}x{height} resolution.")

    rois = extract_roi_from_video(video_path, regions)
    if not rois:
        messagebox.showerror("Error", "No ROIs selected. Detection aborted.")
        return

    ground_truth = get_ground_truth()

    roi_counts = {roi['name']: 0 for roi in rois}
    counted_ids = {roi['name']: set() for roi in rois}
    predicted_counts = []
    true_counts = []

    output_path = f"{target_dir}/Annotated_{video_path.split('/')[-1]}"
    out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = video.read()
        if not ret:
            break

        detections = []

        for roi in rois:
            x_range, y_range = roi['range']
            ROI = frame[y_range[0]:y_range[1], x_range[0]:x_range[1]]

            results = model.predict(ROI, conf=0.25, classes=[0], device='cpu', verbose=False)
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, conf in zip(boxes, confs):
                    xmin, ymin, xmax, ymax = box
                    abs_box = [xmin + x_range[0], ymin + y_range[0], xmax - xmin, ymax - ymin]
                    detections.append(([int(v) for v in abs_box], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            bbox_center = ((ltrb[0] + ltrb[2]) // 2, (ltrb[1] + ltrb[3]) // 2)

            for roi in rois:
                poly = np.array(roi['polygon'], np.int32)
                if cv2.pointPolygonTest(poly, bbox_center, False) >= 0:
                    if track_id not in counted_ids[roi['name']]:
                        counted_ids[roi['name']].add(track_id)
                        roi_counts[roi['name']] += 1

        for idx, roi in enumerate(rois):
            cv2.putText(frame, f'{roi["name"]}: {roi_counts[roi["name"]]}',
                        (30, 30 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)

        if i in ground_truth:
            for region in rois:
                name = region['name']
                true_count = ground_truth[i].get(name, 0)
                true_counts.append(true_count)
                predicted_counts.append(roi_counts[name])

        out_video.write(frame)

    video.release()
    out_video.release()
    print(f"Annotated video saved at {output_path}")

    if true_counts and predicted_counts:
        mae, mape, acc = evaluate_predictions(true_counts, predicted_counts)
        print("\nEvaluation Metrics:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape} %")
        print(f"Counting Accuracy: {acc} %")
    else:
        print("Insufficient data for evaluation.")

    return roi_counts

def browse_video():
    path = filedialog.askopenfilename(title="Select Video")
    entry_video.delete(0, tk.END)
    entry_video.insert(0, path)

def browse_target():
    path = filedialog.askdirectory(title="Select Target Folder")
    entry_target.delete(0, tk.END)
    entry_target.insert(0, path)

def start_detection():
    video_path = entry_video.get()
    target_dir = entry_target.get()
    regions = entry_regions.get().split(',')

    if not video_path or not target_dir or not regions:
        messagebox.showerror("Error", "All fields are required!")
        return

    try:
        results = detect_pedestrians(video_path, target_dir, regions)
        result_text.config(state=tk.NORMAL)
        result_text.insert(tk.END, f"Detection Completed!\n")
        for region, count in results.items():
            result_text.insert(tk.END, f"{region}: {count} pedestrians\n")
        result_text.config(state=tk.DISABLED)
    except Exception as e:
        messagebox.showerror("Error", f"Detection failed: {e}")

root = tk.Tk()
root.title("Pedestrian Detection App")

tk.Label(root, text="Video Path:").grid(row=0, column=0)
entry_video = tk.Entry(root, width=50)
entry_video.grid(row=0, column=1)
tk.Button(root, text="Browse", command=browse_video).grid(row=0, column=2)

tk.Label(root, text="Target Folder:").grid(row=1, column=0)
entry_target = tk.Entry(root, width=50)
entry_target.grid(row=1, column=1)
tk.Button(root, text="Browse", command=browse_target).grid(row=1, column=2)

tk.Label(root, text="Regions (comma-separated):").grid(row=2, column=0)
entry_regions = tk.Entry(root, width=50)
entry_regions.grid(row=2, column=1)

tk.Button(root, text="Start Detection", command=start_detection).grid(row=3, column=1)
result_text = tk.Text(root, height=10, width=60, state=tk.DISABLED)
result_text.grid(row=4, column=0, columnspan=3)

root.mainloop()