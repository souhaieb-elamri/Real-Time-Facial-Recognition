import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox, Toplevel
import threading
import cv2
from PIL import Image, ImageTk
import pyodbc
import datetime
import time
from config import DB_CONN_STR

# Set the root directory to the directory containing this script
root_dir = os.path.dirname(__file__)

# Import callable functions/classes from the same directory
from encode_faces import generate_embeddings
from recognize import FaceRecognizer

# === Global variables ===
cap = None  # Variable to hold the video capture object
camera_running = False  # Flag to control the camera loop
recognizer = None  # Variable to hold the face recognizer instance
camera_source = 0  # Default to local camera (index 0)
last_saved = {}  # Last time each person was saved to the database
SAVE_COOLDOWN_S = 30  # Save a person at most once every 30 seconds

# === Window setup ===
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("1200x800")
root.resizable(True, True)

# === Load background image ===
bg_image_path = os.path.join(root_dir, "ff.jpg")
bg_photo = None
if os.path.exists(bg_image_path):  # Optional background image
    bg_photo = ImageTk.PhotoImage(Image.open(bg_image_path))

# === Load and resize logo image ===
logo_path = os.path.join(root_dir, "wicmic.png")
logo_photo = None
if os.path.exists(logo_path):  # Optional logo
    logo_image = Image.open(logo_path).resize((300, 130), Image.Resampling.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)

# === Canvas for background ===
bg_canvas = tk.Canvas(root, width=1200, height=800)
bg_canvas.pack(fill="both", expand=True)
if bg_photo:
    bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# === Add logo to canvas with adjustable position ===
if logo_photo:
    logo_label = tk.Label(bg_canvas, image=logo_photo, bg="black")
    logo_label.image = logo_photo
    logo_label.place(x=10, y=650)

# === Create frames ===
main_frame = tk.Frame(bg_canvas, bg="#ffffff", bd=2)
main_frame.place(relx=0.03, rely=0.15)

video_label = tk.Label(bg_canvas, bg="black")
video_label.place(relx=0.50, rely=0.05, relwidth=0.47, relheight=0.80)

# === Fonts and Styles ===
label_font = ("Helvetica", 13, "bold")
entry_font = ("Helvetica", 12)
button_font = ("Helvetica", 12, "bold")
button_color = "#2E86C1"
button_fg = "white"

# === Name Input ===
tk.Label(main_frame, text="👤 Name:", font=label_font, bg="white").grid(row=1, column=0, sticky="w", pady=10)
name_entry = tk.Entry(main_frame, font=entry_font, width=25)
name_entry.grid(row=1, column=1, pady=10)

# === Matricule Input ===
tk.Label(main_frame, text="🆔 Matricule:", font=label_font, bg="white").grid(row=2, column=0, sticky="w", pady=10)
matricule_entry = tk.Entry(main_frame, font=entry_font, width=15)
matricule_entry.grid(row=2, column=1, pady=10)

# === Number of Images ===
tk.Label(main_frame, text="📸 Number of Images:", font=label_font, bg="white").grid(row=3, column=0, sticky="w", pady=10)
num_entry = tk.Entry(main_frame, font=entry_font, width=10)
num_entry.grid(row=3, column=1, pady=10)

# === Status Label ===
status_label = tk.Label(main_frame, text="", font=entry_font, bg="white", fg="black")
status_label.grid(row=4, column=0, columnspan=2, pady=10)

# === Camera Selection Window ===
def select_camera():
    global camera_source
    camera_window = Toplevel(root)
    camera_window.title("Select Camera")
    camera_window.geometry("400x350")  # Increased height to accommodate new fields
    camera_window.configure(bg="white")
    camera_window.transient(root)
    camera_window.grab_set()

    tk.Label(
        camera_window,
        text="Select Camera Source",
        font=("Arial", 12),
        bg="white"
    ).pack(pady=10)

    camera_var = tk.StringVar(value="local")

    tk.Radiobutton(
        camera_window,
        text="Local Camera",
        variable=camera_var,
        value="local",
        bg="white",
        font=("Arial", 10)
    ).pack(anchor=tk.W, padx=20, pady=5)

    tk.Radiobutton(
        camera_window,
        text="IP Camera",
        variable=camera_var,
        value="ip",
        bg="white",
        font=("Arial", 10)
    ).pack(anchor=tk.W, padx=20, pady=5)

    tk.Radiobutton(
        camera_window,
        text="Secret Camera",
        variable=camera_var,
        value="secret",
        bg="white",
        font=("Arial", 10)
    ).pack(anchor=tk.W, padx=20, pady=5)

    # Frame for camera inputs
    input_frame = tk.Frame(camera_window, bg="white")
    input_frame.pack(padx=20, pady=5, fill=tk.X)

    # IP Camera URL
    ip_label = tk.Label(input_frame, text="IP Camera URL (e.g., http://192.168.x.x:port/video.mjpg)", bg="white", font=("Arial", 10))
    ip_label.pack(anchor=tk.W)
    ip_camera_var = tk.StringVar(value="")
    ip_camera_entry = tk.Entry(input_frame, textvariable=ip_camera_var, width=40)
    ip_camera_entry.pack(fill=tk.X, pady=5)

    # Secret Camera Credentials
    secret_frame = tk.Frame(input_frame, bg="white")
    secret_frame.pack(fill=tk.X, pady=5)

    tk.Label(secret_frame, text="Username:", bg="white", font=("Arial", 10)).pack(anchor=tk.W)
    username_var = tk.StringVar(value="")
    username_entry = tk.Entry(secret_frame, textvariable=username_var, width=40)
    username_entry.pack(fill=tk.X, pady=2)

    tk.Label(secret_frame, text="Password:", bg="white", font=("Arial", 10)).pack(anchor=tk.W)
    password_var = tk.StringVar(value="")
    password_entry = tk.Entry(secret_frame, textvariable=password_var, width=40, show="*")
    password_entry.pack(fill=tk.X, pady=2)

    def confirm():
        global cap, camera_source
        if camera_var.get() == "local":
            camera_source = 0
            if cap and cap.isOpened():
                cap.release()
            cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                status_label.config(text="Connected to local camera")
                camera_window.destroy()
            else:
                status_label.config(text="Failed to connect to local camera")
        elif camera_var.get() == "ip" and ip_camera_var.get().strip():
            camera_source = ip_camera_var.get().strip()
            if cap and cap.isOpened():
                cap.release()
            cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                status_label.config(text="Connected to IP camera")
                camera_window.destroy()
            else:
                status_label.config(text="Failed to connect to IP camera")
        elif camera_var.get() == "secret" and ip_camera_var.get().strip():
            # Use HTTP URL with credentials if provided, otherwise use the URL as-is
            base_url = ip_camera_var.get().strip()
            if username_var.get().strip() and password_var.get().strip():
                # Insert credentials into the URL
                if base_url.startswith("http://"):
                    base_url = f"http://{username_var.get().strip()}:{password_var.get().strip()}@{base_url.split('://')[-1]}"
                elif base_url.startswith("rtsp://"):
                    base_url = f"rtsp://{username_var.get().strip()}:{password_var.get().strip()}@{base_url.split('://')[-1]}"
                else:
                    # Assume HTTP if no protocol specified
                    base_url = f"http://{username_var.get().strip()}:{password_var.get().strip()}@{base_url}"
            camera_source = base_url
            print(f"Attempting to connect to URL: {camera_source}")  # Debug print
            if cap and cap.isOpened():
                cap.release()
            cap = cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                status_label.config(text="Connected to secret camera")
                camera_window.destroy()
            else:
                status_label.config(text="Failed to connect to secret camera: Check URL, credentials, or camera settings")
                print(f"Connection failed for URL: {camera_source}")  # Debug print
        else:
            status_label.config(text="Please select a camera or enter valid URL/credentials")
        root.update()

    tk.Button(camera_window, text="Confirm", command=confirm, font=("Arial", 10), bg=button_color, fg=button_fg).pack(pady=20)

# === Button Commands ===
def start_capture():
    name = name_entry.get().strip()
    matricule = matricule_entry.get().strip()
    num = num_entry.get().strip()
    if not name or not matricule or not num.isdigit() or not matricule.isdigit() or len(matricule) > 8:
        messagebox.showerror("Error", "Please enter a valid name, a numeric matricule (max 8 digits), and number of images.")
        return
    num = int(num)
    print(f"Running capture.py in directory: {root_dir} with name={name}, matricule={matricule}, num={num}")
    subprocess.run([sys.executable, os.path.join(root_dir, "capture.py"), name, matricule, str(num)], cwd=root_dir)
    status_label.config(text=f"Capture complete for '{name}' (Matricule: {matricule}) with {num} images!")

def start_encode():
    def task():
        generate_embeddings()
        root.after(0, lambda: messagebox.showinfo("Done", "✅ Embeddings updated!"))
        root.after(0, lambda: status_label.config(text="Embeddings updated!"))
    threading.Thread(target=task, daemon=True).start()
    status_label.config(text="Encoding faces...")

def start_recognition():
    global recognizer, camera_running
    if recognizer is None:
        embeddings_path = os.path.join(root_dir, "embeddings.pkl")
        recognizer = FaceRecognizer(embeddings_path)
    camera_running = True
    status_label.config(text="Recognition started...")
    threading.Thread(target=run_recognition, daemon=True).start()

def stop_recognition():
    global camera_running
    camera_running = False
    status_label.config(text="Recognition stopped.")

def run_recognition():
    global cap
    cap = cv2.VideoCapture(camera_source)  # Use the selected camera source
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while camera_running:
        ret, frame = cap.read()
        if ret:
            frame = recognizer.process_frame(frame)
            # Save detected face to SQL Server
            faces = recognizer.app.get(frame)
            for face in faces:
                name, matricule, score = recognizer.recognize_face(face.embedding)
                if name != "Unknown":
                    key = (name, matricule)
                    if time.time() - last_saved.get(key, 0) < SAVE_COOLDOWN_S:
                        continue
                    try:
                        with pyodbc.connect(DB_CONN_STR) as conn:
                            cursor = conn.cursor()
                            current_time = datetime.datetime.now()  # ✅ FIXED
                            cursor.execute("INSERT INTO Persons (Name, Matricule, RegistrationDate) VALUES (?, ?, ?)", (name, matricule, current_time))
                            conn.commit()
                            last_saved[key] = time.time()
                            status_label.config(text=f"Person '{name}' (Matricule: {matricule}) detected and saved at {current_time}")
                    except pyodbc.Error as e:
                        messagebox.showerror("Error", f"Failed to save {name} to SQL Server: {e}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.config(image=imgtk)
            video_label.image = imgtk
            root.update()
            cv2.waitKey(1)
        else:
            messagebox.showerror("Error", "Webcam failed.")
            break
    cap.release()

# === Buttons ===
tk.Button(main_frame, text="Select Camera", font=button_font, bg="#3498db", fg=button_fg,
          width=20, pady=10, command=select_camera).grid(row=5, column=0, columnspan=2, pady=10)
tk.Button(main_frame, text="Capture Face", font=button_font, bg=button_color, fg=button_fg,
          width=20, pady=10, command=start_capture).grid(row=6, column=0, columnspan=2, pady=10)
tk.Button(main_frame, text="Encode Faces", font=button_font, bg="#27AE60", fg=button_fg,
          width=20, pady=10, command=start_encode).grid(row=7, column=0, columnspan=2, pady=10)
tk.Button(main_frame, text="Start Recognition", font=button_font, bg="#C0392B", fg=button_fg,
          width=20, pady=10, command=start_recognition).grid(row=8, column=0, columnspan=2, pady=10)
tk.Button(main_frame, text="Stop Recognition", font=button_font, bg="#F1C40F", fg=button_fg,
          width=20, pady=10, command=stop_recognition).grid(row=9, column=0, columnspan=2, pady=10)

# === Main loop ===
root.mainloop()