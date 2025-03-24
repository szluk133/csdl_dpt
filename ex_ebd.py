import os
import torchaudio
import mysql.connector
import json
from speechbrain.pretrained import SpeakerRecognition

# Cấu hình MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="admin",
    password="123456",
    database="btl_emd"
)
cursor = conn.cursor()

# Load mô hình ECAPA-TDNN
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# Hàm trích xuất embedding
def get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embedding = model.encode_batch(signal).squeeze().tolist()
    return embedding

# Hàm lưu vào MySQL
def insert_embedding(file_path, duration, embedding):
    query = "INSERT INTO voice_samples (file_path, duration, embedding) VALUES (%s, %s, %s)"
    cursor.execute(query, (file_path, duration, json.dumps(embedding)))
    conn.commit()

# Duyệt tất cả file trong thư mục
def process_directory(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Chỉ xử lý file WAV
            file_path = os.path.join(folder_path, filename)
            duration = torchaudio.info(file_path).num_frames / torchaudio.info(file_path).sample_rate
            embedding = get_embedding(file_path)
            insert_embedding(file_path, duration, embedding)
            print(f"✅ Đã lưu {filename} vào MySQL")

# Chạy chương trình
if __name__ == "__main__":
    folder = "E:/NAM_4_KY_2_2025/CSDL_ĐPT/filtered_audio"  # Thư mục chứa file âm thanh
    process_directory(folder)
    print("🎯 Hoàn thành!")
