import argparse
import cv2
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import ExerciseSession


def draw_status(frame, result):
    debug = result.get("debug", {})
    cv2.putText(frame, f"Reps: {result['rep_count']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Stage: {result['stage']}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {result['last_status']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, result["feedback"], (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if "main_angle" in debug:
        cv2.putText(
            frame,
            f"Elbow: {int(debug['main_angle'])}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    y = 215
    for rep in result.get("history", [])[-3:]:
        detail = rep.get("detail", "-")
        cv2.putText(
            frame,
            f"Rep {rep['rep']}: {detail}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        y += 24


def parse_args():
    parser = argparse.ArgumentParser(description="Run push-up detection from webcam without website.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0")
    parser.add_argument("--target-reps", type=int, default=15, help="Target reps for live session.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    return parser.parse_args()


def main():
    args = parse_args()
    session = ExerciseSession("push_up")
    session.reset(target_reps=args.target_reps)

    cap = cv2.VideoCapture(args.camera)
    cap.set(3, args.width)
    cap.set(4, args.height)

    if not cap.isOpened():
        raise RuntimeError("Kamera tidak terbuka")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = session.process_frame(frame, draw_pose=True)
        output_frame = result["frame"]
        draw_status(output_frame, result)
        cv2.imshow("Pushup Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if result["completed"]:
            break

    print("\n=== HASIL ===")
    for rep in session.get_history():
        print(f"Rep {rep['rep']} - {rep['status']}")
        print(f"  Detail: {rep['detail']}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
