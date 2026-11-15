"""Render a few parts of the held-out recording with pilot-model detections."""

import subprocess
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "data" / "boss-zen-heart" / "experiments" / "weak-v1"
VIDEO = PROJECT / "dist" / "boss-trial-recordings" / "gpu-boss-trial-20261109-154051.mp4"
FFMPEG = PROJECT.parent / "android-icon-clicker" / "build" / "video-deps" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v9.0.2.exe"
OUTPUT = RUN / "recognition-replay.mp4"


def main():
    if OUTPUT.exists():
        raise SystemExit(f"Output already exists; preserve it: {OUTPUT}")
    model = YOLO(str(RUN / "weights" / "best.pt"))
    video = cv2.VideoCapture(str(VIDEO))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")
    fps = video.get(cv2.CAP_PROP_FPS)
    width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    encoder = subprocess.Popen([
        str(FFMPEG), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-n", str(OUTPUT),
    ], stdin=subprocess.PIPE)
    count = 0
    try:
        for start, end in ((33, 39), (45, 52), (82, 89)):
            video.set(cv2.CAP_PROP_POS_FRAMES, round(start * fps))
            for _ in range(round((end - start) * fps)):
                ok, frame = video.read()
                if not ok:
                    raise RuntimeError(f"Video ended unexpectedly at {start}s")
                result = model.predict(frame, imgsz=640, conf=0.25, device=0, quantize=16, verbose=False)[0]
                for box, cls, confidence in zip(result.boxes.xyxy.cpu().tolist(),
                                                result.boxes.cls.cpu().tolist(),
                                                result.boxes.conf.cpu().tolist()):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 190, 255), 2)
                    label = f"{model.names[int(cls)]} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 5, 18)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 190, 255), 2)
                cv2.putText(frame, "OFFLINE REPLAY - DETECTION ONLY", (15, height - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                encoder.stdin.write(frame.tobytes())
                count += 1
    finally:
        video.release()
        encoder.stdin.close()
        return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"Video encoder failed: {return_code}")
    print(f"Rendered {count} frames at {fps:g} fps: {OUTPUT}")


if __name__ == "__main__":
    main()
