import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch


from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device

def plot_one_box(x1y1x2y2, img, label=None, color=(255, 0, 0), line_thickness=None):
    """绘制矩形框"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # 线厚度
    c1, c2 = (int(x1y1x2y2[0]), int(x1y1x2y2[1])), (int(x1y1x2y2[2]), int(x1y1x2y2[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # 字体大小
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3  # 确保文本不会超出图片边界
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # 文本背景框
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

class YOLOv5DetectorApp(QMainWindow):
    def __init__(self, weights_path, model, names):
        super().__init__()
        self.model = model
        self.names = names
        self.weights_path = weights_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLOv5 图像检测')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 600, 400)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton('选择图片', self)
        self.select_button.setGeometry(50, 500, 200, 50)
        self.select_button.clicked.connect(self.select_image)

    def select_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "",
                                                    "All Files (*);;Image Files (*.png *.jpg *.jpeg)", options=options)
        if image_path:
            self.detect_objects(image_path)

    def detect_objects(self, image_path):
        img0 = cv2.imread(image_path)  # 读取图像

        # 处理图像通道和格式
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # 转换颜色通道
        img_resized = cv2.resize(img, (640, 640))  # 确保图像的大小符合模型输入的要求

        # 转换图像格式
        img_tensor = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
        img_tensor = torch.from_numpy(img_tensor).to('cpu')  # 转换为PyTorch张量
        img_tensor = img_tensor.float()  # uint8 to fp32
        img_tensor /= 255.0  # 图像归一化到[0, 1]

        # 推理
        pred = self.model(img_tensor, augment=False)[0]

        # 应用 NMS
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        # 处理检测结果
        for i, det in enumerate(pred):  # 检测结果遍历
            if len(det):
                # 将坐标映射回原图
                det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=(255, 0, 0), line_thickness=3)

        # 显示图片
        img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
        height, width, channel = img0_rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(img0_rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.image_label.setPixmap(pixmap.scaledToWidth(600))

if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象

    # 获取权重文件的路径
    weights_path = './runs/train/exp/weights/best.pt'  # 权重文件路径
    config_path = './models/yolov5s.yaml'  # 模型配置文件路径
    data_path = './data/my_data.yaml'  # 数据集配置文件路径

    # 加载模型
    device = select_device('cpu')
    model = attempt_load(weights_path)  # 加载模型和权重，不需要 map_location
    stride = int(model.stride.max())  # 模型的stride
    imgsz = check_img_size(640, s=stride)  # 检查图像大小，640是默认的图像大小
    if hasattr(model, 'names'):
        names = model.module.names if hasattr(model, 'module') else model.names  # 从模型获取类别名
    else:
        names = ['item_{:g}'.format(i) for i in range(1, model.nc + 1)]

    # 创建窗口并显示
    window = YOLOv5DetectorApp(weights_path, model, names)
    window.show()

    sys.exit(app.exec_())  # 应用程序退出管理
