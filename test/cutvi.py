import subprocess

def cut_video_ffmpeg(input_file, output_file, start_sec, end_sec):
    # Tính thời lượng (duration) vì FFmpeg dùng tham số -t là độ dài đoạn cắt
    duration = end_sec - start_sec
    
    # Lệnh ffmpeg: -ss (bắt đầu), -t (thời lượng), -c copy (giữ nguyên gốc, ko encode lại)
    command = [
        'ffmpeg',
        '-ss', str(start_sec),
        '-t', str(duration),
        '-i', input_file,
        '-c', 'copy', # Cực nhanh vì chỉ copy luồng dữ liệu
        output_file,
        '-y' # Ghi đè nếu file đầu ra đã tồn tại
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"--- Cắt video thành công bằng FFmpeg! ---")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy FFmpeg: {e}")

# Sử dụng
cut_video_ffmpeg("../rsrc/v10_te2.mp4", "../rsrc/comp.mp4", 0, 5)