import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis
import pickle
import torch

def generate_embeddings():
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device_name = "CPU" if not cuda_available else torch.cuda.get_device_name(0)
    print("CUDA available:", cuda_available)
    print("Running on:", device_name)

    # Initialize InsightFace with GPU if available, otherwise CPU
    providers = ['CUDAExecutionProvider'] if cuda_available else ['CPUExecutionProvider']
    app = FaceAnalysis(providers=providers)
    app.prepare(ctx_id=0 if cuda_available else -1, det_size=(640, 640))  # GPU ctx_id=0, CPU ctx_id=-1

    KNOWN_FACE_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
    embeddings = {}

    for filename in os.listdir(KNOWN_FACE_DIR):
        if filename.lower().endswith((".jpg", ".png")):
            path = os.path.join(KNOWN_FACE_DIR, filename)
            img = cv2.imread(path)
            if img is None:
                print(f"[!] Failed to load {filename}")
                continue
            faces = app.get(img)

            if faces:
                emb = faces[0].embedding
                # Extract name and matricule (e.g., "souhaieb_12345_1.jpg" → name="souhaieb", matricule="12345")
                parts = filename.split('_')
                name = parts[0]
                matricule = parts[1] if len(parts) > 1 else "N/A"
                embeddings[(name, matricule)] = emb
                print(f"[✓] Processed {filename} → {name} (Matricule: {matricule})")
            else:
                print(f"[!] No face detected in {filename}")

    # Save to embeddings.pkl
    with open("embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

    print("✅ embeddings.pkl generated for all users.")

if __name__ == "__main__":
    generate_embeddings()