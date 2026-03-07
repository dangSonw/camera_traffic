import datetime

class Log:
    def __init__(self, log_path="../logs/terminal.txt"):
        # Lấy đường dẫn file log từ tham số đầu vào
        self.log_path = log_path

    def _write(self, message):
        # Ghi message ra file log, tạo file nếu chưa có
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception as e:
            print(f"[Log] Error writing log: {e}")

    def log(self, *args, sep=" ", end="\n"):
        msg = sep.join(str(a) for a in args) + end.strip("\n")
        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        self._write(f"{timestamp} {msg}")

    def info(self, *args, sep=" ", end="\n"):
        self.log("[INFO]", *args, sep=sep, end=end)

    def warning(self, *args, sep=" ", end="\n"):
        self.log("[WARNING]", *args, sep=sep, end=end)

    def error(self, *args, sep=" ", end="\n"):
        self.log("[ERROR]", *args, sep=sep, end=end)

    def print(self, *args, sep=" ", end="\n"):
        # Ghi log và in ra màn hình
        self.log(*args, sep=sep, end=end)
        print(*args, sep=sep, end=end)