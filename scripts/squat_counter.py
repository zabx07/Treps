import cv2
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import ExerciseSession


MAX_REPS = 14


def draw_status(frame, result):
    debug = result.get("debug", {})
    cv2.putText(frame, f"Reps: {result['rep_count']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Stage: {result['stage']}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {result['last_status']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, result["feedback"], (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if "main_angle" in debug:
        cv2.putText(
            frame,
            f"Knee: {int(debug['main_angle'])}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )


def main():
    session = ExerciseSession("squat")
    session.reset(target_reps=MAX_REPS)

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        raise RuntimeError("Kamera tidak terbuka")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = session.process_frame(frame, draw_pose=True)
        output_frame = result["frame"]
        draw_status(output_frame, result)
        cv2.imshow("Squat Detection", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:
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
