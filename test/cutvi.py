import subprocess
import os

def cut_video_ffmpeg(input_file, output_file, start_sec, end_sec):
    # 1. Kiểm tra file đầu vào có tồn tại không
    if not os.path.exists(input_file):
        print(f"Lỗi: Không tìm thấy file gốc tại {input_file}")
        return

    # 2. Đảm bảo file đầu ra không trùng tên file đầu vào (Tránh mất dữ liệu gốc)
    if os.path.abspath(input_file) == os.path.abspath(output_file):
        print("Lỗi: File đầu ra không được trùng tên với file đầu vào!")
        return

    duration = end_sec - start_sec
    
    # Lệnh FFmpeg tinh chỉnh
    command = [
        'ffmpeg',
        '-y',               # Ghi đè file ĐẦU RA nếu đã tồn tại
        '-ss', str(start_sec), # Bắt đầu từ giây thứ...
        '-i', input_file,   # File gốc
        '-t', str(duration),# Thời lượng cắt
        '-c', 'copy',       # Copy luồng dữ liệu (không encode lại - cực nhanh)
        '-avoid_negative_ts', 'make_zero', # Giúp video bắt đầu đúng từ 0:00
        output_file
    ]
    
    try:
        # Sử dụng stderr=subprocess.PIPE để bắt lỗi chi tiết nếu cần
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"--- Cắt video thành công! ---")
        print(f"File lưu tại: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy FFmpeg: {e.stderr}")

# --- SỬ DỤNG ---
# Ví dụ: Cắt từ giây 0 đến giây 5
cut_video_ffmpeg("../rsrc/ngay_nct___.mp4", "comp_nct_ngay.mp4", 45, 60)