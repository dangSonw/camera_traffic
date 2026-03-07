class Communication:

    def __init__(self, port="/dev/ttyUSB0", baudrate=115200, timeout=1):
        try:
            import serial
            self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        except Exception as e:
            # Chỉ cảnh báo, không dừng chương trình, không raise, không exit
            print(f"[Communication] Warning: Could not open serial port: {e}. Continuing without serial communication.")
            self.ser = None

    def send_data(self, data):
        """
        Gửi dữ liệu (str hoặc bytes) qua UART. Nếu không có serial, bỏ qua.
        """
        if self.ser is not None:
            try:
                if isinstance(data, str):
                    data = data.encode()
                self.ser.write(data)
            except Exception as e:
                print(f"[Communication] Error sending data: {e}")
        # Nếu không có serial, bỏ qua không in gì, không dừng chương trình

    def receive_data(self, size=1024):
        """
        Nhận dữ liệu từ UART, trả về bytes. Nếu không có serial, trả về b"".
        """
        if self.ser is not None:
            try:
                data = self.ser.read(size)
                return data
            except Exception as e:
                print(f"[Communication] Error receiving data: {e}")
                return b""
        # Nếu không có serial, trả về b"" không in gì, không dừng chương trình
        return b""