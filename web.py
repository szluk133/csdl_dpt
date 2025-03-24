import os
import json
import tempfile
import shutil
import numpy as np
import torchaudio
import mysql.connector
from flask import Flask, request, render_template, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
from speechbrain.pretrained import SpeakerRecognition

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CACHE_FOLDER'] = 'audio_cache'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Giới hạn kích thước file 16MB
app.secret_key = 'your_secret_key'  # Thêm secret key cho session

# Tạo thư mục cần thiết
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Kết nối MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="admin",
        password="123456",
        database="btl_emd"
    )

# Tải mô hình ECAPA-TDNN
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

# Hàm trích xuất embedding từ file âm thanh
def get_embedding(audio_path):
    signal, fs = torchaudio.load(audio_path)
    embedding = model.encode_batch(signal).squeeze().tolist()
    return embedding

# Hàm tính toán độ tương đồng cosine
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Hàm tìm file âm thanh tương tự nhất
def find_similar_audio(embedding, count=3):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Lấy tất cả các embedding từ cơ sở dữ liệu
    cursor.execute("SELECT file_path, duration, embedding FROM voice_samples")
    rows = cursor.fetchall()
    
    # Tính toán độ tương đồng với mỗi embedding
    similarities = []
    for row in rows:
        db_embedding = json.loads(row['embedding'])
        similarity = cosine_similarity(embedding, db_embedding)
        similarities.append({
            'file_path': row['file_path'],
            'duration': row['duration'],
            'similarity': similarity
        })
    
    # Sắp xếp theo độ tương đồng (cao nhất đầu tiên)
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Đóng kết nối
    cursor.close()
    conn.close()
    
    # Trả về các kết quả hàng đầu
    return similarities[:count]

# Trang chủ HTML
@app.route('/')
def index():
    return render_template('index.html')

# Tải lên và xử lý
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and file.filename.endswith('.wav'):
        # Lưu file đã tải lên
        filename = secure_filename(file.filename)
        
        # Lưu file vào thư mục uploads thay vì thư mục tạm để có thể phát lại
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Lưu đường dẫn file input vào cache để phát lại
        input_cache_path = os.path.join(app.config['CACHE_FOLDER'], f"input_{filename}")
        shutil.copy2(filepath, input_cache_path)
        
        # Lưu tên file vào session để hiển thị và phát lại
        session['input_filename'] = f"input_{filename}"
        
        # Lấy embedding cho file đã tải lên
        try:
            embedding = get_embedding(filepath)
            
            # Tìm các file tương tự
            similar_files = find_similar_audio(embedding)
            
            # Xử lý kết quả
            results = []
            for item in similar_files:
                file_path = item['file_path']
                file_name = os.path.basename(file_path)
                
                # Tạo bản sao của file âm thanh trong thư mục cache
                cache_path = os.path.join(app.config['CACHE_FOLDER'], file_name)
                try:
                    shutil.copy2(file_path, cache_path)
                except Exception as e:
                    print(f"Lỗi khi sao chép file: {str(e)}")
                
                results.append({
                    'file_path': file_path,
                    'file_name': file_name,
                    'duration': item['duration'],
                    'similarity': item['similarity'],
                    'audio_url': f"/audio/{file_name}"
                })
            
            # Không xóa file input để có thể phát lại
            # Trả về template với kết quả và thông tin file input
            return render_template('index.html', 
                results=results, 
                input_filename=session.get('input_filename'),
                input_audio_url=f"/audio/{session.get('input_filename')}")
            
        except Exception as e:
            return f"Lỗi khi xử lý file: {str(e)}"
    
    return redirect(request.url)

# Route để phục vụ các file âm thanh
@app.route('/audio/<filename>')
def serve_audio(filename):
    cache_path = os.path.join(app.config['CACHE_FOLDER'], filename)
    
    # Kiểm tra xem file có trong cache không
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype='audio/wav')
    
    # Nếu là file input
    if filename.startswith('input_'):
        original_filename = filename[6:]  # Bỏ prefix "input_"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        if os.path.exists(upload_path):
            # Sao chép vào cache
            try:
                shutil.copy2(upload_path, cache_path)
                return send_file(cache_path, mimetype='audio/wav')
            except Exception as e:
                return f"Lỗi khi truy cập file: {str(e)}", 500
    
    # Nếu không có trong cache, lấy từ database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT file_path FROM voice_samples WHERE file_path LIKE %s", (f'%{filename}',))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result:
        original_path = result['file_path']
        if os.path.exists(original_path):
            # Sao chép vào cache
            try:
                shutil.copy2(original_path, cache_path)
                return send_file(cache_path, mimetype='audio/wav')
            except Exception as e:
                return f"Lỗi khi truy cập file: {str(e)}", 500
        else:
            return "File không tồn tại trên hệ thống", 404
    else:
        return "Không tìm thấy file âm thanh trong cơ sở dữ liệu", 404

if __name__ == '__main__':
    app.run(debug=True)