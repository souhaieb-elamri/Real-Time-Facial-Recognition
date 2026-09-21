import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import torch

class FaceRecognizer:
    def __init__(self, embeddings_path="embeddings.pkl"):
        # === Check CUDA availability ===
        cuda_available = torch.cuda.is_available()
        print("CUDA available:", cuda_available)
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
        print("Current device:", device_name)

        # === Load embeddings ===
        with open(embeddings_path, "rb") as f:
            self.known_faces = pickle.load(f)

        # === Initialize InsightFace with appropriate provider ===
        if cuda_available:
            providers = ['CUDAExecutionProvider']  # Use GPU if available
            ctx_id = 0
        else:
            providers = ['CPUExecutionProvider']  # Fall back to CPU if no GPU
            ctx_id = -1  # CPU context ID

        self.app = FaceAnalysis(providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def recognize_face(self, embedding, threshold=0.5):
        best_match = None
        best_score = -1

        for (name, matricule), known_emb in self.known_faces.items():
            score = self.cosine_similarity(embedding, known_emb)
            if score > best_score:
                best_score = score
                best_match = (name, matricule)

        if best_score > threshold:
            return best_match[0], best_match[1], best_score
        else:
            return "Unknown", "N/A", best_score

    def process_frame(self, frame):
        faces = self.app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            name, matricule, score = self.recognize_face(face.embedding)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} (Mat: {matricule}) {int(score * 100)}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame

if __name__ == "__main__":
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognizer.process_frame(frame)
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()