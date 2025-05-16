import os
import requests

# API untuk menerima gambar 
UPLOAD_URL = 'https://api-classify.smartfarm.id/upload-image'  # <- tambahkan https://

# Dapatkan path absolut dari direktori tempat file ini berada
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path absolut ke folder 'segmented' (yang berisi gambar cabai, ganti 'segmented' sama nama folder tempat gambar cabai di raspi)
LOCAL_FOLDER = os.path.join(ROOT_DIR, 'segmented')

# Ambil semua file gambar di folder segmented
for filename in os.listdir(LOCAL_FOLDER):
    filepath = os.path.join(LOCAL_FOLDER, filename)
    
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            files = {'file': (filename, f)}
            try:
                response = requests.post(UPLOAD_URL, files=files)
                print(f'{filename}: {response.status_code} - {response.text}')
            except Exception as e:
                print(f'Error uploading {filename}: {e}')
