import cv2
import time
import urllib.request
import urllib.error
from base64 import b64encode

# ===== CẤU HÌNH CAMERA DAHUA =====
CAMERA_IP = '192.168.50.107'
USERNAME = 'admin'
PASSWORD = 'abcd1234'
RTSP_URL = f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:554/cam/realmonitor?channel=1&subtype=0"

# ===== HÀM GỬI LỆNH CGI =====
def send_dahua_command(cgi_path):
    """Gửi lệnh CGI đến camera Dahua với Digest Authentication"""
    try:
        url = f"http://{USERNAME}:{PASSWORD}@{CAMERA_IP}{cgi_path}"
        req = urllib.request.Request(url)
        
        # Tạo password manager cho Digest Auth
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url, USERNAME, PASSWORD)
        
        # Thử Digest Auth trước
        auth_handler = urllib.request.HTTPDigestAuthHandler(password_mgr)
        opener = urllib.request.build_opener(auth_handler)
        
        response = opener.open(req, timeout=5)
        result = response.read().decode()
        return result
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

# ===== CÁC LỆNH PTZ CHO DAHUA =====
def zoom_in(speed=5):
    """Zoom in - speed từ 1-8"""
    cmd = f"/cgi-bin/ptz.cgi?action=start&channel=1&code=ZoomTele&arg1=0&arg2={speed}&arg3=0"
    result = send_dahua_command(cmd)
    if result:
        print(f"Zoom In (speed={speed})")
    return result

def zoom_out(speed=5):
    """Zoom out - speed từ 1-8"""
    cmd = f"/cgi-bin/ptz.cgi?action=start&channel=1&code=ZoomWide&arg1=0&arg2={speed}&arg3=0"
    result = send_dahua_command(cmd)
    if result:
        print(f"Zoom Out (speed={speed})")
    return result

def focus_near(speed=5):
    """Focus gần - speed từ 1-8"""
    cmd = f"/cgi-bin/ptz.cgi?action=start&channel=1&code=FocusNear&arg1=0&arg2={speed}&arg3=0"
    result = send_dahua_command(cmd)
    if result:
        print(f"Focus Near (speed={speed})")
    return result

def focus_far(speed=5):
    """Focus xa - speed từ 1-8"""
    cmd = f"/cgi-bin/ptz.cgi?action=start&channel=1&code=FocusFar&arg1=0&arg2={speed}&arg3=0"
    result = send_dahua_command(cmd)
    if result:
        print(f"Focus Far (speed={speed})")
    return result

def stop_ptz():
    """Dừng tất cả chuyển động PTZ"""
    cmd = f"/cgi-bin/ptz.cgi?action=stop&channel=1"
    result = send_dahua_command(cmd)
    if result:
        print("PTZ Stopped")
    return result

def set_absolute_position(pan=0, tilt=0, zoom=0):
    """
    Di chuyển đến vị trí tuyệt đối
    pan: 0-3600 (độ * 10)
    tilt: 0-900 (độ * 10)
    zoom: 0-100 (%)
    """
    cmd = f"/cgi-bin/ptz.cgi?action=start&channel=1&code=PositionABS&arg1={pan}&arg2={tilt}&arg3={zoom}"
    result = send_dahua_command(cmd)
    if result:
        print(f"Move to Position: Pan={pan}, Tilt={tilt}, Zoom={zoom}")
    return result

def get_ptz_status():
    """Lấy trạng thái PTZ hiện tại"""
    cmd = "/cgi-bin/ptz.cgi?action=getStatus&channel=1"
    result = send_dahua_command(cmd)
    if result:
        print(f"PTZ Status: {result}")
    return result

def auto_focus():
    """Bật auto focus"""
    cmd = "/cgi-bin/ptz.cgi?action=start&channel=1&code=AutoFocus&arg1=0&arg2=0&arg3=0"
    result = send_dahua_command(cmd)
    if result:
        print("Auto Focus Enabled")
    return result

def get_device_info():
    """Lấy thông tin thiết bị"""
    cmd = "/cgi-bin/magicBox.cgi?action=getDeviceType"
    result = send_dahua_command(cmd)
    if result:
        print(f"Device Info: {result}")
    return result

def get_zoom_focus_status():
    """Lấy thông tin zoom và focus hiện tại"""
    cmd = "/cgi-bin/devVideoInput.cgi?action=getFocusStatus&channel=1"
    result = send_dahua_command(cmd)
    if result:
        print(f"Zoom/Focus Status: {result}")
    return result

# ===== IN THÔNG TIN CAMERA =====
print(f"{'='*60}")
print("THÔNG TIN CAMERA DAHUA")
print(f"{'='*60}")
print(f"IP: {CAMERA_IP}")
print(f"Username: {USERNAME}")
print(f"RTSP URL: {RTSP_URL}")

print(f"\n--- Device Information ---")
get_device_info()

print(f"\n--- PTZ Status ---")
get_ptz_status()

print(f"\n--- Zoom/Focus Status ---")
get_zoom_focus_status()

# ===== TEST CÁC LỆNH ĐIỀU KHIỂN =====
print(f"\n{'='*60}")
print("TEST ĐIỀU KHIỂN PTZ")
print(f"{'='*60}")

print("\n1. Test Zoom In (2 giây)...")
zoom_in(speed=5)
time.sleep(2)
stop_ptz()

print("\n2. Test Zoom Out (2 giây)...")
zoom_out(speed=5)
time.sleep(2)
stop_ptz()

print("\n3. Test Focus Near (1 giây)...")
focus_near(speed=3)
time.sleep(1)
stop_ptz()

print("\n4. Test Focus Far (1 giây)...")
focus_far(speed=3)
time.sleep(1)
stop_ptz()

print("\n5. Test Auto Focus...")
auto_focus()
time.sleep(2)

print("\n6. Test Absolute Position (Pan=0, Tilt=0, Zoom=50)...")
set_absolute_position(pan=0, tilt=0, zoom=50)
time.sleep(2)

# ===== HIỂN THỊ VIDEO VỚI ĐIỀU KHIỂN REAL-TIME =====
print(f"\n{'='*60}")
print("HIỂN THỊ VIDEO VỚI ĐIỀU KHIỂN PTZ")
print(f"{'='*60}")

cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print(f"KHÔNG MỞ ĐƯỢC RTSP: {RTSP_URL}")
    exit()

print("Kết nối RTSP thành công!")
print("\n=== PHÍM ĐIỀU KHIỂN ===")
print("  z: Zoom In (giữ phím)")
print("  x: Zoom Out (giữ phím)")
print("  f: Focus Near (giữ phím)")
print("  g: Focus Far (giữ phím)")
print("  a: Auto Focus")
print("  h: Home Position (Pan=0, Tilt=0, Zoom=0)")
print("  1-9: Zoom nhanh đến mức (1=10%, 5=50%, 9=90%)")
print("  q: Thoát")

zoom_active = False
focus_active = False
zoom_level = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Mất kết nối RTSP!")
        break
    
    # Resize nếu quá lớn (tối ưu cho Raspberry Pi)
    height, width = frame.shape[:2]
    if width > 1280:
        scale = 1280 / width
        frame = cv2.resize(frame, (1280, int(height * scale)))
    
    # Hiển thị thông tin
    cv2.putText(frame, f"Zoom: {zoom_level}%", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    info1 = "z:Zoom+ | x:Zoom- | f:FocusNear | g:FocusFar | a:AutoFocus"
    cv2.putText(frame, info1, (10, height - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    info2 = "h:Home | 1-9:QuickZoom | q:Quit"
    cv2.putText(frame, info2, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Dahua Camera - PTZ Control', frame)
    
    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('q'):
        break
    
    elif key == ord('z'):  # Zoom in
        if not zoom_active:
            zoom_in(speed=5)
            zoom_active = True
    
    elif key == ord('x'):  # Zoom out
        if not zoom_active:
            zoom_out(speed=5)
            zoom_active = True
    
    elif key == ord('f'):  # Focus near
        if not focus_active:
            focus_near(speed=4)
            focus_active = True
    
    elif key == ord('g'):  # Focus far
        if not focus_active:
            focus_far(speed=4)
            focus_active = True
    
    elif key == ord('a'):  # Auto focus
        auto_focus()
        zoom_active = False
        focus_active = False
    
    elif key == ord('h'):  # Home position
        set_absolute_position(pan=0, tilt=0, zoom=0)
        zoom_level = 0
        zoom_active = False
        focus_active = False
    
    elif ord('1') <= key <= ord('9'):  # Quick zoom 1-9
        zoom_level = (key - ord('0')) * 10
        set_absolute_position(pan=0, tilt=0, zoom=zoom_level)
        zoom_active = False
        focus_active = False
    
    else:
        # Thả phím -> dừng PTZ
        if zoom_active or focus_active:
            stop_ptz()
            zoom_active = False
            focus_active = False

stop_ptz()
cap.release()
cv2.destroyAllWindows()
print("\nHoàn tất!")
