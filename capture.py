import cv2
import os
import mediapipe as mp
import numpy as np
import sys  # Added to handle command-line arguments

# === CONFIG ===
frame_width = 1280
frame_height = 720
box_size = 400

def is_face_centered(bbox, capture_zone):
    x, y, w, h = capture_zone
    fx = int(bbox[0] * frame_width)
    fy = int(bbox[1] * frame_height)
    fw = int(bbox[2] * frame_width)
    fh = int(bbox[3] * frame_height)
    return x < fx < x + w and y < fy < y + h and (fx + fw) < (x + w) and (fy + fh) < (y + h)

def capture_images(name, matricule, num_images):
    save_dir = os.path.join(os.path.dirname(__file__), "known_faces")
    os.makedirs(save_dir, exist_ok=True)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    box_x = (frame_width - box_size) // 2
    box_y = (frame_height - box_size) // 2
    capture_zone = (box_x, box_y, box_size, box_size)

    font = cv2.FONT_HERSHEY_SIMPLEX
    count = 0

    print(f"[INFO] Saving images for '{name}' (Matricule: {matricule}) in {save_dir}/")
    print("📸 Center your face and press SPACE to capture. ESC to quit.")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam failed.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 3)
            ready = False

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                for lm in face_landmarks.landmark:
                    cx = int(lm.x * frame_width)
                    cy = int(lm.y * frame_height)
                    cv2.circle(frame, (cx, cy), 1, (0, 255, 255), -1)

                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]

                if is_face_centered(bbox, capture_zone):
                    cv2.putText(frame, "✅ Face OK - Press SPACE", (30, 60), font, 0.9, (0, 255, 0), 2)
                    ready = True
                else:
                    cv2.putText(frame, "🔴 Center your face", (30, 60), font, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "🛑 No face detected", (30, 60), font, 0.9, (0, 0, 255), 2)

            cv2.putText(frame, f"{count}/{num_images} images", (30, frame_height - 30), font, 0.8, (255, 255, 255), 2)

            cv2.imshow("Face Capture", frame)
            key = cv2.waitKey(1)

            if key % 256 == 27:
                print("❌ Exit requested.")
                break
            elif key % 256 == 32:
                if ready:
                    img_path = os.path.join(save_dir, f"{name}_{matricule}_{count+1}.jpg")
                    cv2.imwrite(img_path, frame)
                    print(f"✅ Saved: {img_path}")
                    count += 1
                    if count >= num_images:
                        print("🎉 Done capturing.")
                        break
                else:
                    print("⚠️ Face not ready.")

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python capture.py <name> <matricule> <number_of_images>")
        sys.exit(1)
    name = sys.argv[1]  # Name from GUI
    matricule = sys.argv[2]  # Matricule from GUI
    num_images = int(sys.argv[3])  # Number of images from GUI
    capture_images(name, matricule, num_images)