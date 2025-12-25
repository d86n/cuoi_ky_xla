import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox, QButtonGroup, QFrame
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush
from PySide6.QtCore import Qt, QTimer, QPoint

try:
    from ultralytics import YOLO
except ImportError:
    print("Cài đặt: pip install ultralytics")
    sys.exit(1)

# --- CẤU HÌNH ---
REAL_USB_WIDTH_MM = 9.0 

# --- WIDGET HIỂN THỊ ẢNH TÙY CHỈNH (QUAN TRỌNG) ---
class InteractiveDisplay(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True) # Bật theo dõi chuột liên tục
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed #ccc; background: #fff;")
        
        self.pixmap_orig = None     # Ảnh gốc sạch
        self.detections = []        # Danh sách kết quả từ YOLO
        self.hovered_item = None    # Linh kiện đang được trỏ chuột vào
        self.scale_ratio = 1.0      # Tỷ lệ co giãn ảnh

    def update_data(self, cv_img, detections):
        """Nhận ảnh và dữ liệu từ Main Window"""
        self.detections = detections
        
        # Convert CV2 -> QPixmap
        h, w, ch = cv_img.shape
        bytes_per_line = 3 * w
        q_img = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.pixmap_orig = QPixmap.fromImage(q_img)
        self.update() # Vẽ lại

    def mouseMoveEvent(self, event):
        """Xử lý khi di chuột"""
        if not self.detections or not self.pixmap_orig: return

        # Tính tỷ lệ scale hiện tại giữa ảnh gốc và khung hiển thị
        img_w = self.pixmap_orig.width()
        lbl_w = self.width()
        
        # Nếu ảnh được scale fit center (KeepAspectRatio)
        # Ta cần tính toán kỹ để map tọa độ chuột sang tọa độ ảnh
        pixmap_scaled = self.pixmap_orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Offset (khoảng trắng thừa ra nếu ảnh không full khung)
        offset_x = (self.width() - pixmap_scaled.width()) / 2
        offset_y = (self.height() - pixmap_scaled.height()) / 2
        
        self.scale_ratio = self.pixmap_orig.width() / pixmap_scaled.width()

        # Tọa độ chuột trên ảnh gốc
        mouse_x = (event.position().x() - offset_x) * self.scale_ratio
        mouse_y = (event.position().y() - offset_y) * self.scale_ratio

        # Tìm linh kiện gần chuột nhất (trong bán kính 20px)
        found = None
        min_dist = 30 * self.scale_ratio # Bán kính tìm kiếm
        
        for item in self.detections:
            cx, cy = item['cx'], item['cy']
            dist = ((mouse_x - cx)**2 + (mouse_y - cy)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                found = item
        
        if found != self.hovered_item:
            self.hovered_item = found
            self.update() # Trigger hàm paintEvent vẽ lại

    def paintEvent(self, event):
        """Vẽ chồng lớp thông tin lên ảnh"""
        super().paintEvent(event) # Vẽ ảnh nền (được setPixmap từ trước)
        
        if not self.pixmap_orig: return

        # Chúng ta vẽ đè lên Label bằng QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Tính toán lại geometry của ảnh đã scale đang hiện trên label
        scaled_pixmap = self.pixmap_orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = (self.width() - scaled_pixmap.width()) // 2
        offset_y = (self.height() - scaled_pixmap.height()) // 2
        scale = scaled_pixmap.width() / self.pixmap_orig.width()

        # 1. VẼ CÁC CHẤM/KHUNG MỜ CHO TẤT CẢ LINH KIỆN (Trạng thái tĩnh)
        for item in self.detections:
            # Map tọa độ từ ảnh gốc ra màn hình
            sx = int(item['cx'] * scale) + offset_x
            sy = int(item['cy'] * scale) + offset_y
            
            # Chỉ vẽ chấm nhỏ màu xanh lá (Rất sạch mắt)
            painter.setBrush(QBrush(QColor(255, 0, 0, 150))) # Màu xanh, hơi trong suốt
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(sx, sy), 4, 4)

        # 2. VẼ THÔNG TIN CHI TIẾT KHI HOVER (Trạng thái động)
        if self.hovered_item:
            item = self.hovered_item
            sx = int(item['cx'] * scale) + offset_x
            sy = int(item['cy'] * scale) + offset_y
            
            # Vẽ vòng tròn highlight quanh điểm đó
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawEllipse(QPoint(sx, sy), 10, 10)
            
            # Tạo nội dung Text
            text = f"{item['label']}\n{item['pos_mm']}"
            
            # Vẽ hộp nền cho text (Tooltip)
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            fm = painter.fontMetrics()
            rect_w = fm.horizontalAdvance(item['pos_mm']) + 20
            rect_h = 40
            
            # Vị trí hộp text (tránh bị tràn ra ngoài màn hình)
            tx, ty = sx + 15, sy - 15
            
            # Vẽ hộp đen mờ
            painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(tx, ty, rect_w, rect_h, 5, 5)
            
            # Vẽ chữ trắng
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(tx + 10, ty + 15, item['label'])
            painter.setPen(QColor(0, 255, 255)) # Màu cyan cho tọa độ
            painter.drawText(tx + 10, ty + 32, item['pos_mm'])

        painter.end()


# --- CỬA SỔ CHÍNH ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Inspection Pro (Hover Mode)")
        self.setMinimumSize(1200, 800)
        
        self.model = None
        self.load_model()
        
        self.is_camera_running = False
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_camera)

        self.setup_ui()

    def load_model(self):
        try:
            self.model = YOLO('best.pt')
            print("Model loaded.")
        except:
            self.model = YOLO('yolov8n.pt')

    def setup_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        # SIDEBAR
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background: #222; color: #fff;") # Dark mode cho ngầu
        sb = QVBoxLayout(sidebar)
        
        lbl = QLabel("<h2>🎛 CONTROL</h2>")
        lbl.setStyleSheet("color: #00ff00;")
        sb.addWidget(lbl)
        
        sb.addWidget(QLabel("Mốc: USB-C (9mm)"))
        sb.addSpacing(20)

        # Buttons
        btn_css = """
            QPushButton { background: #444; border: none; padding: 10px; color: white; text-align: left; }
            QPushButton:hover { background: #555; }
            QPushButton:checked { background: #007bff; }
        """
        
        self.btn_img = QPushButton("📸 Chế độ Ảnh")
        self.btn_img.setCheckable(True); self.btn_img.setChecked(True); self.btn_img.setStyleSheet(btn_css)
        self.btn_img.clicked.connect(self.stop_camera)
        
        self.btn_cam = QPushButton("🎥 Chế độ Camera")
        self.btn_cam.setCheckable(True); self.btn_cam.setStyleSheet(btn_css)
        self.btn_cam.clicked.connect(self.start_camera)

        grp = QButtonGroup(self)
        grp.addButton(self.btn_img); grp.addButton(self.btn_cam)
        sb.addWidget(self.btn_img); sb.addWidget(self.btn_cam)
        
        sb.addSpacing(10)
        self.btn_open = QPushButton("📂 Mở Ảnh")
        self.btn_open.setStyleSheet("background: #28a745; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_open.clicked.connect(self.open_image)
        sb.addWidget(self.btn_open)

        sb.addStretch()
        sb.addWidget(QLabel("LOG CHI TIẾT:"))
        self.txt_log = QTextEdit()
        self.txt_log.setStyleSheet("background: #111; color: #0f0; font-family: Consolas;")
        sb.addWidget(self.txt_log)

        self.display = InteractiveDisplay()
        
        layout.addWidget(sidebar)
        layout.addWidget(self.display, stretch=1)

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Img (*.jpg *.png)")
        if fname:
            img = cv2.imread(fname)
            self.run_ai(img)

    def start_camera(self):
        self.cap = cv2.VideoCapture(1)
        self.is_camera_running = True
        self.timer.start(30)
        self.btn_open.setEnabled(False)

    def stop_camera(self):
        self.timer.stop()
        if self.cap: self.cap.release()
        self.is_camera_running = False
        self.btn_open.setEnabled(True)

    def process_camera(self):
        ret, frame = self.cap.read()
        if ret: self.run_ai(frame)

    def run_ai(self, img_orig):
        # 1. Detect
        results = self.model(img_orig, conf=0.25, verbose=False)
        
        # 2. Xử lý dữ liệu (Tính toán mm, tọa độ)
        h, w = img_orig.shape[:2]
        detections = [] # List chứa thông tin sạch để vẽ sau
        mm_per_px = None
        
        # Tìm USB trước để lấy scale
        for r in results:
            for box in r.boxes:
                label = self.model.names[int(box.cls[0])]
                if label in ['usb_port', 'usb']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    mm_per_px = REAL_USB_WIDTH_MM / max(x2-x1, y2-y1)
                    # Thêm USB vào list nhưng đánh dấu là mốc
                    detections.append({
                        'label': 'REF: USB', 'cx': (x1+x2)//2, 'cy': (y1+y2)//2, 
                        'pos_mm': '0,0 (Gốc)', 'is_ref': True
                    })
                    break
            if mm_per_px: break

        # Duyệt các linh kiện còn lại
        counts = {}
        log_txt = ""
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = self.model.names[int(box.cls[0])]
                cx, cy = (x1+x2)//2, (y1+y2)//2
                
                # Tính mm
                pos_str = f"({cx}, {cy})px"
                if mm_per_px:
                    rx, ry = cx * mm_per_px, cy * mm_per_px
                    pos_str = f"({rx:.1f}, {ry:.1f})mm"
                
                # Lưu vào list data (Không vẽ cứng lên ảnh nữa!)
                detections.append({
                    'label': label,
                    'cx': cx, 'cy': cy,
                    'pos_mm': pos_str,
                    'is_ref': False
                })
                
                counts[label] = counts.get(label, 0) + 1

        # 3. Cập nhật giao diện
        # Gửi ảnh GỐC SẠCH (img_orig) và danh sách DATA sang widget hiển thị
        # Widget đó sẽ tự lo việc vẽ chồng lớp
        
        # Cập nhật Text Log bên trái
        log_txt = "--- SỐ LƯỢNG ---\n"
        for k,v in counts.items(): log_txt += f"{k}: {v}\n"
        self.txt_log.setText(log_txt)
        
        # Cập nhật hình ảnh
        # Chú ý: Ta vẽ USB Reference cứng lên ảnh một chút cho dễ nhìn mốc
        img_draw = img_orig.copy()
        if mm_per_px:
             # Vẽ mỗi cái khung USB thôi cho đỡ rối
             for item in detections:
                 if item.get('is_ref'):
                     cv2.circle(img_draw, (item['cx'], item['cy']), 5, (255, 0, 0), -1)
                     break
        
        self.display.update_data(img_draw, detections)
        self.display.setPixmap(self.display.pixmap_orig.scaled(self.display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())