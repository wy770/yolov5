import sys

import cv2
import torch
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

import cv2
import torch
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self, model, device, conf_thres=0.25, iou_thres=0.45):
        super().__init__()
        self.model = model
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img).to(self.device)
                img = img.permute(2, 0, 1)  # 转换形状为 (C, H, W)
                img = img.half() if self.model.fp16 else img.float()
                img /= 255.0
                img = img.unsqueeze(0)  # 添加批次维度 (1, C, H, W)

                pred = self.model(img)
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
                for det in pred:
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{self.model.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)

                # 将图像从BGR转换为RGB以显示在GUI中
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img)
            else:
                break
        cap.release()

    def stop(self):
        self.running = False
        self.wait()

def plot_one_box(xyxy, img, color=(128, 128, 128), label=None, line_thickness=3):
    tl = line_thickness
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv5 实时检测")
        self.setGeometry(100, 100, 800, 600)

        self.device = 'cpu'
        self.model = self.model_load(weights="./best.pt", device=self.device)

        self.image_label = QLabel(self)
        self.image_label.resize(800, 600)

        self.start_button = QPushButton("开始检测", self)
        self.start_button.clicked.connect(self.start_video)

        self.stop_button = QPushButton("停止检测", self)
        self.stop_button.clicked.connect(self.stop_video)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.video_thread = None

    @torch.no_grad()
    def model_load(self, weights="", device='', half=False, dnn=False):
        device = select_device(device)
        half &= device.type != 'cpu'
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        half &= pt and device.type != 'cpu'
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成，模型信息如上!")
        return model

    def start_video(self):
        self.video_thread = VideoThread(self.model, self.device)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def stop_video(self):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None

    def update_image(self, qt_img):
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        self.stop_video()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())



# python .\Video_UI.py
