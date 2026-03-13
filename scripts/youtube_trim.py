import cv2
import os

BASE_DIR = "/data1/vailab02_dir/Classification_DB/tmp_vid"

timeline = {
    "Arson041_x264.mp4": (40.0, 120.0),
    "Assault042_x264.mp4": (0.0, 100.0),
    "fdd_roomfire41.mp4": (14.0, 59.0),
    "Robbery146_x264.mp4": (0.0, 30.0),
}


def trim_video(path, start_sec, end_sec, out_path):
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= end_frame:
            break

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


for video, (start, end) in timeline.items():
    in_path = os.path.join(BASE_DIR, video)

    name, ext = os.path.splitext(video)
    out_name = f"{name}_trimmed{ext}"
    out_path = os.path.join(BASE_DIR, out_name)

    print(f"Trimming {video}: {start} -> {end}")

    trim_video(in_path, start, end, out_path)

print("Done.")