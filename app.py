from __future__ import annotations
import shutil
import base64
import os
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

from model_service import MnistModelService


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-mnist-secret-key")

    base_dir = Path(__file__).resolve().parent
    feedback_dir = base_dir / "dataset_feedback"
    correct_dir = feedback_dir / "correct"
    incorrect_dir = feedback_dir / "incorrect"
    pending_dir = base_dir / "static" / "pending"

    correct_dir.mkdir(parents=True, exist_ok=True)
    incorrect_dir.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)

    # инициализация модели один раз при старте сервера
    model_service = MnistModelService(
        weights_path=base_dir / "network2.json",
        image_size=(28, 28),
        auto_invert=True,
    )

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(silent=True) or {}
        data_url = payload.get("image_data", "")

        if not data_url or "," not in data_url:
            return jsonify({"error": "Пустое изображение с холста."}), 400

        try:
            # отделение заголовока data:image/png;base64, от самих байтов
            _, encoded = data_url.split(",", 1)
            raw_bytes = base64.b64decode(encoded)

            pil_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
            prediction = model_service.predict(pil_img)

            # сохранение рисунка во временную папку до нажатия «верно» / «не верно»
            pending_filename = f"{uuid.uuid4().hex}.png"
            pil_img.save(pending_dir / pending_filename, format="PNG")

            return jsonify({
                "digit": prediction["digit"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "pending_filename": pending_filename,
            })
        except Exception as exc:
            return jsonify({"error": f"Ошибка обработки рисунка: {exc}"}), 500

    @app.route("/feedback", methods=["POST"])
    def feedback():
        payload = request.get_json(silent=True) or {}
        pending_filename = str(payload.get("pending_filename", "")).strip()
        is_correct = bool(payload.get("is_correct"))
        predicted_digit = str(payload.get("predicted_digit", "unknown")).strip()
        true_digit = str(payload.get("true_digit", "")).strip()

        source_path = pending_dir / Path(pending_filename).name
        if not pending_filename or not source_path.exists():
            return jsonify({"error": "Рисунок для оценки не найден или уже сохранён."}), 404

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        valid_digits = {str(i) for i in range(10)}

        if is_correct:
            target_dir = correct_dir
            actual_label = true_digit if true_digit in valid_digits else predicted_digit
            message = "Сохранено в dataset_feedback/correct"
        else:
            target_dir = incorrect_dir
            actual_label = true_digit if true_digit in valid_digits else "unknown"
            message = "Сохранено в dataset_feedback/incorrect"

        out_name = f"pred_{predicted_digit}_true_{actual_label}_{timestamp}_{short_id}.png"
        shutil.move(str(source_path), str(target_dir / out_name))

        return jsonify({"status": "ok", "message": message, "saved_file": out_name})

    return app


app = create_app()

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host=host, port=port, debug=debug)