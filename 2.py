
import sys
from tkinter.font import names

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device\

# -*- coding: utf-8 -*-
# 导入库
import shutil
import PyQt5.QtCore
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import threading
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import sys
import shutil
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont
from PyQt5.QtCore import Qt
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
class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('YOLOV5学生课堂行为检测图片识别')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        self.model = None
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.initUI()

    def initUI(self):
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)

        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)

        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)

        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)

        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")

        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        self.addTab(img_detection_widget, '图片检测')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))

    def upload_img(self):
        fileName, _ = QFileDialog.getOpenFileName(self, '选择文件', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))

    def detect_img(self):
        if not self.model:
            self.model = attempt_load(weights="./runs/train/exp/weights/best.pt", device=self.device)

        source = self.img2predict
        imgsz = [640, 640]
        conf_thres = self.conf_thres
        iou_thres = self.iou_thres
        max_det = 1000
        classes = None
        agnostic_nms = False
        augment = False
        visualize = False
        line_thickness = 3
        hide_labels = False
        hide_conf = False
        half = False
        dnn = False

        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            device = select_device(self.device)
            stride = int(self.model.stride.max())
            imgsz = check_img_size(imgsz, s=stride)

            im0 = cv2.imread(source)
            img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)  # 转换颜色通道
            img_resized = cv2.resize(img, (640, 640))  # 确保图像的大小符合模型输入的要求

            img_tensor = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
            img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
            img_tensor = torch.from_numpy(img_tensor).to(self.device)  # 转换为PyTorch张量
            img_tensor = img_tensor.float()  # uint8 to fp32
            img_tensor /= 255.0  # 图像归一化到[0, 1]

            pred = self.model(img_tensor, augment=False)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{int(cls)} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=(255, 0, 0), line_thickness=line_thickness)

            im0 = cv2.resize(im0, (0, 0), fx=0.5, fy=0.5)
            cv2.imwrite("images/tmp/single_result.jpg", im0)
            self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

def plot_one_box(x, img, color=(128, 128, 128), label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
