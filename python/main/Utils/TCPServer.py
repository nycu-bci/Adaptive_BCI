import socket
import threading
import time

_server = None  # 目前的 TCPServer (供 LSL 連上後啟動 watchdog)
_lsl_connected = False


def on_lsl_connected():
    """LSL 連上後呼叫：啟動 Unity 連線 watchdog (每個 process 一次)"""
    global _lsl_connected
    _lsl_connected = True
    if _server is not None:
        _server.start_watchdog()


class TCPServer:
    def __init__(self, host='0.0.0.0', port=50007, on_message=None):
        """
        on_message: callback function(msg:str) -> None
        """
        self.host = host
        self.port = port
        self.on_message = on_message
        self.clients = []
        self.stop_event = threading.Event()
        self.server_socket = None
        self.thread = None
        self.ever_connected = False
        self.watchdog_thread = None

    def start(self):
        global _server
        _server = self
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        print(f"🚀 TCP Server started on {self.host}:{self.port}")
        if _lsl_connected:
            self.start_watchdog()

    def start_watchdog(self, check_s=1.0, grace_s=3.0, repeat_s=5.0, log=print):
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return
        self.watchdog_thread = threading.Thread(
            target=self._watchdog, args=(check_s, grace_s, repeat_s, log), daemon=True)
        self.watchdog_thread.start()

    def _watchdog(self, check_s, grace_s, repeat_s, log):
        no_client_since = None
        last_warn = None
        while not self.stop_event.is_set():
            now = time.monotonic()
            if self.clients:
                no_client_since = last_warn = None
            else:
                if no_client_since is None:
                    no_client_since = now
                if now - no_client_since >= grace_s and (last_warn is None or now - last_warn >= repeat_s):
                    last_warn = now
                    what = "Connection to unity lost" if self.ever_connected else "Waiting for Unity connection"
                    log(f"⚠️ WARNING: {what}")
            time.sleep(check_s)

    def stop(self):
        self.stop_event.set()
        if self.server_socket:
            self.server_socket.close()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        print("🛑 TCP Server stopped")

    def _run_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()

        while not self.stop_event.is_set():
            try:
                self.server_socket.settimeout(1.0)
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            except Exception:
                continue

    def _handle_client(self, conn, addr):
        print(f"✅ Client connected from {addr}")
        self.clients.append(conn)
        self.ever_connected = True
        try:
            while not self.stop_event.is_set():
                data = conn.recv(1024)
                if not data:
                    break
                msg = data.decode('utf-8').strip()
                print(f"📥 Received from {addr}: {msg}")
                if self.on_message:
                    self.on_message(msg, self)
        except Exception as e:
            print(f"⚠️ Connection error from {addr}: {e}")
        finally:
            conn.close()
            if conn in self.clients:
                self.clients.remove(conn)
            print(f"❌ Disconnected: {addr}")

    def broadcast(self, message: str):
        """廣播訊息給所有連線中的 client"""
        dead_clients = []
        for conn in self.clients:
            try:
                conn.sendall(message.encode('utf-8'))
            except Exception as e:
                print(f"⚠️ 無法傳送給 {conn.getpeername()}: {e}")
                dead_clients.append(conn)
        # 移除斷線的 client
        for conn in dead_clients:
            self.clients.remove(conn)
