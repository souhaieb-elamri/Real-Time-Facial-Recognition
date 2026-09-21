# Real-Time Facial Recognition (InsightFace)

Desktop application that enrolls people from a webcam, computes face embeddings with **InsightFace**, recognizes them in real time from a local or IP camera, and logs each recognized person in **SQL Server**.

## Features

- **Guided face capture** with MediaPipe: a target box and live feedback ("Face OK" / "Center your face") before each photo
- **Face embeddings** computed with InsightFace, on GPU (CUDA) when available and on CPU otherwise
- **Real-time recognition** by cosine similarity, with a decision threshold of 0.5 and the match score shown on screen
- **Local or IP camera** support (HTTP or RTSP streams)
- **SQL Server logging** of the recognized person (name, ID, timestamp), at most once every 30 seconds per person
- **Tkinter interface** to capture, encode and recognize without using the command line

## Pipeline

1. **Capture** (`capture.py`): saves photos named `name_matricule_n.jpg`.
2. **Encode** (`encode_faces.py`): computes one embedding per person and saves `embeddings.pkl`.
3. **Recognize** (`recognize.py`): compares each detected face with the known embeddings.
4. **Interface** (`gui_app.py`): runs the steps above and writes recognized people to SQL Server.

## Results

Accuracy: `[95% on N people / M test images. Describe how you measured it]`

## Setup

```bash
pip install -r requirements.txt
```

1. Create the database table (example):

```sql
CREATE TABLE Persons (
    Id INT IDENTITY PRIMARY KEY,
    Name NVARCHAR(100),
    Matricule NVARCHAR(20),
    RegistrationDate DATETIME
);
```

2. Edit `config.py` with your SQL Server address.
3. Run the application:

```bash
python gui_app.py
```

4. In the interface: **Select Camera**, then **Capture Face**, then **Encode Faces**, then **Start Recognition**.

## Repository structure

```
gui_app.py         # Tkinter interface and database logging
capture.py         # Face capture with MediaPipe guidance
encode_faces.py    # InsightFace embeddings generation
recognize.py       # Real-time recognition (cosine similarity)
config.py          # SQL Server connection settings (placeholders)
requirements.txt
```

## Privacy

Face photos (`known_faces/`) and `embeddings.pkl` contain biometric data and are **not** included in this repository. Only enroll people who gave their consent.

## Author

**Souhaieb Elamri**, Electrical Engineering student (ENSIT), Master in Advanced Robotics & AI
[LinkedIn](https://linkedin.com/in/souhaieb-elamri) | [Portfolio](https://souhaieb-elamri.github.io/Portfolio/)
