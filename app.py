import os
import base64
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
from flask_cors import CORS
import mimetypes

load_dotenv()

app = Flask(__name__)

FRONTEND_ORIGIN = "http://localhost:3000"
CORS(app, supports_credentials=True, origins=[FRONTEND_ORIGIN, "chrome-extension://*", "moz-extension://*", "http://127.0.0.1:3000", "null"])

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
        print("ðŸ•” Gambar otomatis dipindahkan setelah klasifikasi.")

    timer = threading.Timer(0, delayed_move)  # 0 detik
    timer.start()

    return jsonify({
        "classification_result": results_list,
        "message": "Klasifikasi selesai. Gambar akan dipindahkan otomatis dalam 5 menit, atau Anda bisa trigger manual.",
        "auto_move_time": "0 detik"
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

    # Jika tidak ada tanggal, ambil tanggal dan waktu terbaru
    if not tanggal:
        cursor.execute("""
            SELECT tanggal, waktu FROM chili_predictions_v1
            ORDER BY tanggal DESC, waktu DESC LIMIT 1
        """)
        result = cursor.fetchone()
        if result:
            tanggal = result['tanggal']
            waktu = result['waktu']

    # Bangun query dinamis berdasarkan filter yang ada
    query = """
        SELECT tanggal, waktu, image, pred_class_1, pred_class_2, pred_class_3
        FROM chili_predictions_v1
        WHERE 1=1
    """
    params = []

    if tanggal:
        query += " AND tanggal = %s"
        params.append(tanggal)
    if waktu:
        query += " AND waktu = %s"
        params.append(waktu)
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

    results = []

    for row in rows:
        # Format tanggal
        if isinstance(row['tanggal'], (datetime, date)):
            tanggal_str = row['tanggal'].strftime('%Y-%m-%d')
        else:
            tanggal_str = str(row['tanggal'])

        # Format waktu tergantung tipe
        waktu_raw = row['waktu']
        if isinstance(waktu_raw, timedelta):
            total_seconds = int(waktu_raw.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            waktu_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            waktu_folder = f"{hours:02}-{minutes:02}-{seconds:02}"
        elif isinstance(waktu_raw, time):
            waktu_str = waktu_raw.strftime("%H:%M:%S")
            waktu_folder = waktu_raw.strftime("%H-%M-%S")
        else:
            waktu_str = str(waktu_raw)
            waktu_folder = waktu_str.replace(":", "-")

        # Buat nama folder dan path gambar
        jam_menit = waktu_folder[:5]  # "HH-MM"
        jam, menit = map(int, jam_menit.split('-')) 

        prefixes = []
        prefixes.append(f"{tanggal_str}_{jam:02}-{menit:02}")

        if menit == 59:
            jam_plus = (jam + 1) % 24
            menit_plus = 0
        else:
            jam_plus = jam
            menit_plus = menit + 1

        prefixes.append(f"{tanggal_str}_{jam_plus:02}-{menit_plus:02}")

        image_filename = row['image']

        matching_folders = [f for f in os.listdir(STORAGE_DIR)
                            if any(f.startswith(prefix) for prefix in prefixes)
                            and os.path.isdir(os.path.join(STORAGE_DIR, f))]

        if matching_folders:
            selected_folder = matching_folders[0]
            image_path = os.path.join(STORAGE_DIR, selected_folder, image_filename)
        else:
            image_path = None

        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as img_file:
                    raw_data = img_file.read()
                    encoded = base64.b64encode(raw_data).decode('utf-8')
                    mime_type, _ = mimetypes.guess_type(image_path)
                    if not mime_type:
                        mime_type = "image/jpeg"
                    image_data = f"data:{mime_type};base64,{encoded}"
            except Exception as e:
                print(f"Error encoding {image_path}: {e}")
                image_data = None
        else:
            image_data = None

        results.append({
            "tanggal": tanggal_str,
            "waktu": waktu_str,
            "image": image_filename,
            "image_data": image_data,
            "pred_class_1": row['pred_class_1'],
            "pred_class_2": row['pred_class_2'],
            "pred_class_3": row['pred_class_3']
        })

    cursor.close()
    db.close()

    return jsonify(results)
    
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
    
def get_all_dirs():
    try:
        return [d for d in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, d))]
    except FileNotFoundError:
        return []

def get_week_range(year, month, week):
    first_day = datetime(year, month, 1)
    first_monday = first_day + timedelta(days=(7 - first_day.weekday()) % 7) if first_day.weekday() != 0 else first_day
    start_date = first_monday + timedelta(weeks=week - 1)
    end_date = start_date + timedelta(days=6)
    return start_date.date(), end_date.date()

@app.route("/filter-directories", methods=["POST"])
def filter_directories():
    data = request.get_json() or {}
    tahun = data.get("tahun")
    bulan = data.get("bulan")
    minggu = data.get("minggu")
    tanggal = data.get("tanggal")

    dirs = get_all_dirs()
    matched_dirs = []

    # --- Kondisi 1: JSON kosong, kembalikan direktori 7 hari terakhir ---
    if not data:
        today = datetime.today().date()
        seven_days_ago = today - timedelta(days=6)
        matched_dirs = [
            d for d in dirs
            if any((seven_days_ago + timedelta(days=i)).strftime("%Y-%m-%d") in d for i in range(7))
        ]
        return jsonify({
            "info": "Menampilkan direktori untuk 7 hari terakhir",
            "matched_directories": matched_dirs
        })

    # --- Validasi kombinasi input yang diperbolehkan ---
    fields_provided = {k: v for k, v in {"tahun": tahun, "bulan": bulan, "minggu": minggu, "tanggal": tanggal}.items() if v is not None}

    valid_comb_1 = ("tahun" in fields_provided and "bulan" in fields_provided and "minggu" not in fields_provided and "tanggal" not in fields_provided)
    valid_comb_2 = ("tahun" in fields_provided and "bulan" in fields_provided and "minggu" in fields_provided and "tanggal" not in fields_provided)
    valid_comb_3 = ("tanggal" in fields_provided and len(fields_provided) == 1)

    if not (valid_comb_1 or valid_comb_2 or valid_comb_3):
        return jsonify({
            "error": "Tolong masukkan kombinasi:\n1) tahun dan bulan,\n2) tahun, bulan, dan minggu, atau\n3) tanggal"
        }), 400

    # --- Kondisi 2: hanya tanggal ---
    if valid_comb_3:
        try:
            date_obj = datetime.strptime(tanggal, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Format tanggal tidak valid, gunakan YYYY-MM-DD"}), 400

        matched_dirs = [d for d in dirs if tanggal in d]
        return jsonify({"matched_directories": matched_dirs})

    # --- Kondisi 3: tahun dan bulan (tanpa minggu) ---
    if valid_comb_1:
        try:
            tahun = int(tahun)
            bulan = int(bulan)
            prefix = f"{tahun}-{bulan:02d}"
            matched_dirs = [d for d in dirs if prefix in d]
            return jsonify({"matched_directories": matched_dirs})
        except:
            return jsonify({"error": "Format tahun/bulan tidak valid"}), 400

    # --- Kondisi 4: tahun, bulan, dan minggu ---
    if valid_comb_2:
        try:
            tahun = int(tahun)
            bulan = int(bulan)
            minggu = int(minggu)
            start_week, end_week = get_week_range(tahun, bulan, minggu)
        except Exception:
            return jsonify({"error": "Parameter tahun/bulan/minggu tidak valid"}), 400

        matched_dirs = [
            d for d in dirs
            if any((start_week + timedelta(days=i)).strftime("%Y-%m-%d") in d for i in range(7))
        ]
        return jsonify({"matched_directories": matched_dirs})

    # Fallback (harusnya tidak akan sampai sini karena sudah divalidasi)
    return jsonify({
        "error": "Tolong masukkan kombinasi:\n1) tahun dan bulan,\n2) tahun, bulan, dan minggu, atau\n3) tanggal"
    }), 400
    
    
@app.route('/get-full-image', methods=['POST'])
def get_full_image():
    data = request.get_json(silent=True) or {}
    tanggal = data.get("tanggal")  # format: "YY-MM-DD"
    waktu = data.get("waktu")      # format: "HH:MM:SS"

    if not tanggal or not waktu:
        return jsonify({"error": "Missing 'tanggal' or 'waktu'"}), 400

    try:
        target_dt = datetime.strptime(f"{tanggal} {waktu}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return jsonify({"error": "Invalid date or time format"}), 400

    # Buat nama folder target
    target_folder_name = target_dt.strftime("%Y-%m-%d_%H-%M-%S")

    # Buat daftar nama folder yang dalam rentang ± 1 menit
    toleransi = [target_dt + timedelta(seconds=offset) for offset in range(0, 61)]
    toleransi_names = [dt.strftime("%Y-%m-%d_%H-%M-%S") for dt in toleransi]

    # Cek direktori yang sesuai
    matching_folders = [
        f for f in os.listdir(STORAGE_DIR)
        if f in toleransi_names and os.path.isdir(os.path.join(STORAGE_DIR, f))
    ]

    if not matching_folders:
        return jsonify({"error": "No matching directory found"}), 404

    # Gunakan folder pertama yang ditemukan
    selected_folder = matching_folders[0]
    folder_path = os.path.join(STORAGE_DIR, selected_folder)

    # Cari gambar dengan nama mengandung "full"
    full_images = [
        f for f in os.listdir(folder_path)
        if "full" in f.lower() and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not full_images:
        return jsonify({"error": "No image containing 'full' found"}), 404

    # Ambil gambar pertama
    image_name = full_images[0]
    image_path = os.path.join(folder_path, image_name)

    try:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/jpeg"
            image_data = f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        return jsonify({"error": f"Failed to read image: {str(e)}"}), 500

    return jsonify({
        "image_name": image_name,
        "image_data": image_data
    })
        
# Entry point for WSGI servers
application = app

if __name__ == '__main__':
    app.run(port=5000, debug=True)
