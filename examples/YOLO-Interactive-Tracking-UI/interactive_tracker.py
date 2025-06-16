import time
from typing import Tuple
import cv2
from threading import Thread
from queue import Queue

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
enable_gpu = False
model_file = "yolov8n.pt"
show_fps = True
show_conf = False
save_video = True
video_output_path = "interactive_tracker_output.avi"

conf = 0.3
iou = 0.3
max_det = 20
tracker = "bytetrack.yaml"
track_args = {"persist": True, "verbose": False}
window_name = "Ultralytics YOLO Interactive Tracking"

resize_dims = (640, 480)  # Resize to improve performance
detect_every_n_frames = 2  # Run detection every N frames

# ─────────────────────────────────────────────────────────────
# Initialize
# ─────────────────────────────────────────────────────────────
LOGGER.info("🚀 Initializing model...")
model = YOLO(model_file, task="detect")
if enable_gpu:
    model.to("cuda")

classes = model.names

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

w, h = resize_dims
fps = cap.get(cv2.CAP_PROP_FPS) or 30
vw = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) if save_video else None

selected_object_id = None

# ─────────────────────────────────────────────────────────────
# Frame Reader Thread
# ─────────────────────────────────────────────────────────────
frame_queue = Queue(maxsize=5)


def frame_reader():
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_queue.put(frame)


reader_thread = Thread(target=frame_reader, daemon=True)
reader_thread.start()


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def get_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    return (x1 + x2) // 2, (y1 + y2) // 2


def extend_line_from_edge(mid_x: int, mid_y: int, direction: str, img_shape: Tuple[int, int, int]) -> Tuple[int, int]:
    h, w = img_shape[:2]
    return {"left": (0, mid_y), "right": (w - 1, mid_y), "up": (mid_x, 0), "down": (mid_x, h - 1)}.get(
        direction, (mid_x, mid_y)
    )


def draw_tracking_scope(im, bbox: tuple, color: tuple) -> None:
    x1, y1, x2, y2 = bbox
    mid_top = ((x1 + x2) // 2, y1)
    mid_bottom = ((x1 + x2) // 2, y2)
    mid_left = (x1, (y1 + y2) // 2)
    mid_right = (x2, (y1 + y2) // 2)
    cv2.line(im, mid_top, extend_line_from_edge(*mid_top, "up", im.shape), color, 2)
    cv2.line(im, mid_bottom, extend_line_from_edge(*mid_bottom, "down", im.shape), color, 2)
    cv2.line(im, mid_left, extend_line_from_edge(*mid_left, "left", im.shape), color, 2)
    cv2.line(im, mid_right, extend_line_from_edge(*mid_right, "right", im.shape), color, 2)


# Store latest results for click callback
last_results = [None]


def click_event(event: int, x: int, y: int, flags: int, param) -> None:
    global selected_object_id
    if event == cv2.EVENT_LBUTTONDOWN:
        if last_results[0] and last_results[0].boxes is not None:
            detections = last_results[0].boxes.data
            for track in detections:
                track = track.tolist()
                if len(track) >= 6:
                    x1, y1, x2, y2 = map(int, track[:4])
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        track_id = int(track[4]) if len(track) == 7 else -1
                        selected_object_id = track_id
                        print(f"🔵 TRACKING STARTED: ID {selected_object_id}")


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, click_event)

# ─────────────────────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────────────────────
fps_counter = 0
fps_timer = time.time()
fps_display = 0
frame_count = 0
results = None

while cap.isOpened():
    if frame_queue.empty():
        continue

    frame = frame_queue.get()
    im = cv2.resize(frame, resize_dims)
    frame_count += 1

    run_detection = frame_count % detect_every_n_frames == 0

    if run_detection or results is None:
        try:
            results = model.track(im, conf=conf, iou=iou, max_det=max_det, tracker=tracker, **track_args)
        except Exception as e:
            LOGGER.warning(f"⚠️ Tracking failed: {e}")
            continue

    if results is None or len(results) == 0 or results[0].boxes is None:
        LOGGER.warning("⚠️ No detection results.")
        continue

    last_results[0] = results[0]  # Update for click event
    annotator = Annotator(im)
    detections = results[0].boxes.data

    for track in detections:
        track = track.tolist()
        if len(track) < 6:
            continue
        x1, y1, x2, y2 = map(int, track[:4])
        class_id = int(track[6]) if len(track) >= 7 else int(track[5])
        track_id = int(track[4]) if len(track) == 7 else -1
        color = colors(track_id, True)
        label = f"{classes[class_id]} ID {track_id}" + (f" ({float(track[5]):.2f})" if show_conf else "")

        if track_id == selected_object_id:
            draw_tracking_scope(im, (x1, y1, x2, y2), color)
            center = get_center(x1, y1, x2, y2)
            cv2.circle(im, center, 6, color, -1)
            pulse_radius = 8 + int(4 * abs(time.time() % 1 - 0.5))
            cv2.circle(im, center, pulse_radius, color, 2)
            annotator.box_label([x1, y1, x2, y2], label=f"ACTIVE: TRACK {track_id}", color=color)
        else:
            annotator.box_label([x1, y1, x2, y2], label=label, color=color)

    if show_fps:
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        fps_text = f"FPS: {fps_display}"
        (tw, th), bl = cv2.getTextSize(fps_text, 0, 0.7, 2)
        cv2.rectangle(im, (10 - 5, 25 - th - 5), (10 + tw + 5, 25 + bl), (255, 255, 255), -1)
        cv2.putText(im, fps_text, (10, 25), 0, 0.7, (104, 31, 17), 1, cv2.LINE_AA)

    cv2.imshow(window_name, im)
    if save_video and vw is not None:
        vw.write(im)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        selected_object_id = None

cap.release()
if save_video and vw is not None:
    vw.release()
cv2.destroyAllWindows()
