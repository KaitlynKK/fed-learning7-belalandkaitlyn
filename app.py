from flask import Flask, request, render_template, redirect, url_for
import os
import threading
import subprocess
import base64

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

detection_results = []

@app.route("/", methods=["GET", "POST"])
def index():
    global detection_results
    return render_template("index.html", detections=detection_results)

@app.route("/start-training", methods=["POST"])
def start_training():
    import time

    rounds = request.form.get("rounds")
    clients = request.form.get("clients")
    dataset = request.form.get("dataset")
    data_distribution = request.form.get("data_distribution")

    print(f"[INFO] Starting training: Rounds={rounds}, Clients={clients}, Dataset={dataset}, Dist={data_distribution}")

    threading.Thread(target=lambda: subprocess.run(["python", "server_yolo.py"])).start()
    threading.Thread(target=lambda: subprocess.run(["python", "client_yolo.py"])).start()

    time.sleep(10)

    global detection_results
    detection_results.clear()
    for root, dirs, files in os.walk(OUTPUT_FOLDER):
        for filename in files:
            if filename.endswith(".jpg"):
                full_path = os.path.join(root, filename)
                with open(full_path, "rb") as img_file:
                    b64 = base64.b64encode(img_file.read()).decode("utf-8")
                    detection_results.append(b64)

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
