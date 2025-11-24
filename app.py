from flask import Flask, request, jsonify
import os
import numpy as np
import face_recognition

app = Flask(__name__)

REGISTER_DIR = "register"
os.makedirs(REGISTER_DIR, exist_ok=True)

REGISTER_IMAGE = os.path.join(REGISTER_DIR, "face.jpg")
REGISTER_EMBED = os.path.join(REGISTER_DIR, "face.npy")


# ---------- Extract embedding ----------
def get_embedding(image_path):
    img = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        return None
    return encodings[0]


# ---------- API: Register Face ----------
@app.route("/register", methods=["POST"])
def register_face():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    image = request.files["image"]
    image.save(REGISTER_IMAGE)

    embedding = get_embedding(REGISTER_IMAGE)
    if embedding is None:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    np.save(REGISTER_EMBED, embedding)

    return jsonify({"status": "success", "message": "Face registered"})


# ---------- API: Compare Face ----------
@app.route("/compare", methods=["POST"])
def compare_face():
    if not os.path.exists(REGISTER_EMBED):
        return jsonify({"status": "error", "message": "No registered face found"}), 400

    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image uploaded"}), 400

    temp_img = "temp.jpg"
    request.files["image"].save(temp_img)

    incoming_embed = get_embedding(temp_img)
    if incoming_embed is None:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    registered_embed = np.load(REGISTER_EMBED)

    # Euclidean distance
    distance = np.linalg.norm(incoming_embed - registered_embed)

    THRESHOLD = 0.6  # Good threshold for 1:1 match

    result = "match" if distance < THRESHOLD else "no_match"

    return jsonify({
        "status": "success",
        "result": result,
        "distance": float(distance)
    })


@app.route("/")
def home():
    return "Face Match Server Running"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
