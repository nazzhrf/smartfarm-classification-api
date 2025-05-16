import os
import shutil
import json
from datetime import datetime, date, time
from datetime import timedelta
from flask import Flask, request, jsonify
from decimal import Decimal
from PIL import Image
import threading
import mysql.connector
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Definisi path berbasis direktori root skrip ini
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(ROOT_DIR, 'Temp')
STORAGE_DIR = os.path.join(ROOT_DIR, 'Storage')
MODEL_PATH = os.path.join(ROOT_DIR, 'Model', 'best.pt')
RESULTS_DIR = os.path.join(ROOT_DIR, 'Results')
SCHEDULE_FILE = os.path.join(ROOT_DIR, 'Scheduler/schedule_config.json')


@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    save_path = os.path.join(TEMP_DIR, file.filename)
    file.save(save_path)
    return f'Uploaded {file.filename}', 200

def move_segmented_images_internal():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dest_dir = os.path.join(STORAGE_DIR, timestamp)

    os.makedirs(dest_dir, exist_ok=True)

    moved_files = []
    for filename in os.listdir(TEMP_DIR):
        source_path = os.path.join(TEMP_DIR, filename)
        dest_path = os.path.join(dest_dir, filename)

        if os.path.isfile(source_path):
            shutil.move(source_path, dest_path)
            moved_files.append(filename)

    # Hapus sisa isi (jika ada) tanpa menghapus folder temp itu sendiri
    for leftover in os.listdir(TEMP_DIR):
        path = os.path.join(TEMP_DIR, leftover)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    return {
        "message": "Semua gambar berhasil dipindahkan.",
        "jumlah_file": len(moved_files),
        "folder_baru": dest_dir,
        "files": moved_files
    }

@app.route('/classify', methods=['GET'])
def classify_chili_route():
    model = YOLO(MODEL_PATH)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(TEMP_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return jsonify([{"error": "Tidak ada file gambar di folder temp."}]), 404

    # Koneksi ke database
    db = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )
    cursor = db.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chili_predictions_v1 (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tanggal DATE,
            waktu TIME,
            image VARCHAR(255),
            pred_class_1 VARCHAR(100),
            conf_1 FLOAT,
            pred_class_2 VARCHAR(100),
            conf_2 FLOAT,
            pred_class_3 VARCHAR(100),
            conf_3 FLOAT
        )
    """)

    now = datetime.now()
    tanggal = now.strftime("%Y-%m-%d")
    waktu = now.strftime("%H-%M-%S")

    results_list = []

    for image_file in image_files:
        image_path = os.path.join(TEMP_DIR, image_file)
        img = Image.open(image_path)

        results = model.predict(img, verbose=False)[0]

        names = model.model.names
        print("Class names:", names)
        probs = results.probs.data
        top_indices = probs.argsort(descending=True)[:3]
        print("Top indices:", top_indices)
        print("Top probs:", probs[top_indices])

        pred_classes = [names[int(i)] for i in top_indices]
        confidences = [float(probs[int(i)]) for i in top_indices]

        cursor.execute("""
            INSERT INTO chili_predictions_v1 (
                tanggal, waktu, image,
                pred_class_1, conf_1,
                pred_class_2, conf_2,
                pred_class_3, conf_3
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            tanggal, waktu.replace("-", ":"), image_file,
            pred_classes[0], confidences[0],
            pred_classes[1], confidences[1],
            pred_classes[2], confidences[2]
        ))
        db.commit()

        results_list.append({
            "tanggal": tanggal,
            "waktu": waktu.replace("-", ":"),
            "image": image_file,
            "top3": [
                {"class": pred_classes[0], "confidence": round(confidences[0], 4)},
                {"class": pred_classes[1], "confidence": round(confidences[1], 4)},
                {"class": pred_classes[2], "confidence": round(confidences[2], 4)},
            ]
        })

    cursor.close()
    db.close()

    # Simpan hasil klasifikasi ke file
    output_filename = f"results_{tanggal}_{waktu}.txt"
    output_path = os.path.join(RESULTS_DIR, output_filename)

    with open(output_path, "w") as f:
        json.dump(results_list, f, indent=4)

    # Jadwalkan pemindahan otomatis setelah 5 menit
    def delayed_move():
        move_segmented_images_internal()
        print("ðŸ•” Gambar dipindahkan otomatis 5 menit setelah klasifikasi.")

    timer = threading.Timer(300, delayed_move)  # 5 menit = 300 detik
    timer.start()

    return jsonify({
        "classification_result": results_list,
        "message": "Klasifikasi selesai. Gambar akan dipindahkan otomatis dalam 5 menit, atau Anda bisa trigger manual.",
        "auto_move_time": "5 minutes"
    })
    
def run_classify():
    with app.test_request_context():
        return classify_chili_route()

@app.route('/move', methods=['GET'])
def move_segmented_images_route():
    result = move_segmented_images_internal()
    return jsonify(result)

@app.route('/get-data', methods=['POST'])
def get_data():
    data = request.get_json(silent=True) or {}
    tanggal = data.get('tanggal')
    waktu = data.get('waktu')
    image = data.get('image')
    pred_class_1 = data.get('pred_class_1')
    pred_class_2 = data.get('pred_class_2')
    pred_class_3 = data.get('pred_class_3')

    db = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )

    cursor = db.cursor(dictionary=True)

    # Ambil tanggal dan waktu terbaru jika tidak diberikan
    if not tanggal or not waktu:
        cursor.execute("""
            SELECT tanggal, waktu FROM chili_predictions_v1
            ORDER BY tanggal DESC, waktu DESC LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            if not tanggal:
                tanggal = result['tanggal']
            if not waktu:
                waktu = result['waktu']

    query = """
        SELECT tanggal, waktu, image, pred_class_1, pred_class_2, pred_class_3
        FROM chili_predictions_v1
        WHERE tanggal = %s AND waktu = %s
    """
    params = [tanggal, waktu]

    if image:
        query += " AND image LIKE %s"
        params.append(f"%{image}%")
    if pred_class_1:
        query += " AND pred_class_1 = %s"
        params.append(pred_class_1)
    if pred_class_2:
        query += " AND pred_class_2 = %s"
        params.append(pred_class_2)
    if pred_class_3:
        query += " AND pred_class_3 = %s"
        params.append(pred_class_3)

    query += " ORDER BY waktu DESC LIMIT 30"

    cursor.execute(query, tuple(params))
    rows = cursor.fetchall()

    # Convert tanggal dan waktu ke string supaya JSON bisa serializable dengan baik
    for row in rows:
        if isinstance(row['tanggal'], (datetime, date)):
            row['tanggal'] = row['tanggal'].strftime('%Y-%m-%d')
        if not isinstance(row['waktu'], str):
            row['waktu'] = str(row['waktu'])

    cursor.close()
    db.close()

    return jsonify(rows)

@app.route('/search', methods=['GET'])
def search_data():
    q = request.args.get('q', '').strip()

    db = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )
    cursor = db.cursor(dictionary=True)

    # Hanya ambil kolom yang dibutuhkan (tanpa conf_*)
    query = """
        SELECT tanggal, waktu, image, pred_class_1, pred_class_2, pred_class_3
        FROM chili_predictions_v1
    """
    where_clauses = []
    params = []

    # Daftar kolom string (conf_* tidak termasuk lagi)
    string_fields = ["tanggal", "waktu", "image", "pred_class_1", "pred_class_2", "pred_class_3"]

    if "=" in q:
        field, value = q.split("=", 1)
        field = field.strip()
        value = value.strip()

        if field in string_fields:
            where_clauses.append(f"{field} LIKE %s")
            params.append(f"%{value}%")
        else:
            return jsonify({"error": f"Kolom '{field}' tidak dikenali."}), 400
    else:
        like_term = f"%{q}%"
        for field in string_fields:
            where_clauses.append(f"{field} LIKE %s")
            params.append(like_term)

    if where_clauses:
        query += " WHERE " + " OR ".join(where_clauses)

    query += " ORDER BY tanggal DESC, waktu DESC LIMIT 30"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    # Konversi tipe waktu ke string jika perlu
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (datetime, date, time)):
                row[key] = str(value)
            elif isinstance(value, timedelta):
                row[key] = str(value)
            elif isinstance(value, Decimal):
                row[key] = float(value)

    cursor.close()
    db.close()

    return jsonify(rows)

@app.route('/delete', methods=['POST'])
def delete_data():
    data = request.get_json()

    tanggal = data.get('tanggal')
    waktu = data.get('waktu')
    image = data.get('image')

    if not all([tanggal, waktu, image]):
        return jsonify({"error": "Parameter tanggal, waktu, dan image wajib diisi."}), 400

    db = mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        database=os.getenv("DB_NAME")
    )
    cursor = db.cursor()

    cursor.execute("""
        DELETE FROM chili_predictions_v1
        WHERE tanggal = %s AND waktu = %s AND image = %s
    """, (tanggal, waktu, image))
    db.commit()

    cursor.close()
    db.close()

    return jsonify({"message": "Data berhasil dihapus."})
    
@app.route('/clean-old-dirs', methods=['GET'])
def clean_old_directories():
    # Struktur direktori
    
    deleted_dirs = []
    now = datetime.now()

    for folder_name in os.listdir(STORAGE_DIR):
        folder_path = os.path.join(STORAGE_DIR, folder_name)
        if os.path.isdir(folder_path):
            try:
                folder_date_str = folder_name.split('_')[0]  # e.g., '2023-05-10'
                folder_date = datetime.strptime(folder_date_str, "%Y-%m-%d")
                if (now - folder_date).days > 730:
                    shutil.rmtree(folder_path)
                    deleted_dirs.append(folder_name)
            except Exception as e:
                continue  # Lewati folder dengan format nama yang tidak sesuai

    return jsonify({"deleted_directories": deleted_dirs})

def run_clean_old_dir():
    with app.test_request_context():
        return clean_old_directories()
    
@app.route('/delete-dir', methods=['POST'])
def delete_directory_by_name():
    try:
        data = request.get_json()
        if not data or 'direktori' not in data:
            return jsonify({"error": "JSON harus memiliki field 'direktori'"}), 400

        keyword = data['direktori']

        deleted_dirs = []

        for folder_name in os.listdir(STORAGE_DIR):
            if keyword in folder_name:
                folder_path = os.path.join(STORAGE_DIR, folder_name)
                if os.path.isdir(folder_path):
                    try:
                        shutil.rmtree(folder_path)
                        deleted_dirs.append(folder_name)
                    except Exception as e:
                        continue  # Lewati jika gagal menghapus

        return jsonify({
            "keyword": keyword,
            "deleted_directories": deleted_dirs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
        

@app.route('/update-schedule', methods=['POST']) 
def update_schedule():
    data = request.get_json()
    run_now = data.get("run_now", False)
    data.pop("run_now", None)  # Hapus run_now sebelum disimpan

    with open(SCHEDULE_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    message = "Schedule updated."
    if run_now:
        check_schedule_internal()
        message += " check_schedule() executed."

    return jsonify({"status": "success", "message": message})

@app.route('/check-schedule', methods=['GET'])
def check_schedule():
    return jsonify(check_schedule_internal())


def check_schedule_internal():
    if not os.path.exists(SCHEDULE_FILE):
        return {"status": "error", "message": "No schedule config found."}

    with open(SCHEDULE_FILE, 'r') as f:
        schedule = json.load(f)

    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_weekday = now.strftime("%A")
    current_date = now.strftime("%m-%d")

    triggered_tasks = []

    for task_name, config in schedule.items():
        if "per_day" in config and current_time in config["per_day"]:
            triggered_tasks.append(task_name)
        if "per_week" in config:
            for entry in config["per_week"]:
                day, time = entry.split()
                if day == current_weekday and time == current_time:
                    triggered_tasks.append(task_name)
        if "per_year" in config:
            for entry in config["per_year"]:
                date, time = entry.split()
                if date == current_date and time == current_time:
                    triggered_tasks.append(task_name)

    for task in set(triggered_tasks):
        if task == "classify":
            run_classify()
        elif task == "clean_old_dir":
            run_clean_old_directories()

    return {
        "triggered": list(set(triggered_tasks)),
        "time": current_time
    }
        
# Entry point for WSGI servers
application = app

if __name__ == '__main__':
    app.run(port=5000, debug=True)
