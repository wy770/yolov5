import cv2
import torch
import numpy as np
import base64
from flask import Flask, request, jsonify, send_from_directory
from utils.general import non_max_suppression, scale_coords
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
import time

app = Flask(__name__)

global_device = 'cpu'  # 设备选择为 CPU
weights_path = './best.pt'  # 模型权重路径
model = None

# class names
names = ['hand-raising', 'reading', 'writing', 'normal', 'bowing the head', 'leaning over the table']

# 加载模型
def load_model():
    global model
    global global_device
    global weights_path

    device = select_device(global_device)
    model = DetectMultiBackend(weights_path, device=device)
    model.model.to(device)
    model.model.eval()

@app.route('/')
def index():
    return send_from_directory('.', './index.html')

@app.route('/api/behavior-detection', methods=['POST'])
def behavior_detection():
    global model

    if model is None:
        load_model()

    # 获取图像数据
    data = request.get_json()
    image_data = data['image'].split(',')[1]

    # 解码图像数据
    img_np = np.frombuffer(base64.b64decode(image_data), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # 调整图像大小（降低分辨率）
    img = cv2.resize(img, (320, 320))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_rgb).to(model.device).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    a = time.time()
    with torch.no_grad():
        pred = model.model(img_tensor)
    b = time.time()
    print("推理时间：", b - a)

    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # 处理检测结果
    results = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                cls = int(cls)
                label = f'{model.names[cls]} {conf:.2f}'
                if model.names[cls] == 'hand-raising':
                    # 手在上半部分，扩展手的检测框
                    hand_box = [xyxy[0], xyxy[1], xyxy[2], xyxy[1] + (xyxy[3] - xyxy[1]) // 2]
                    results.append({
                        'label': label,
                        'xyxy': [int(coord) for coord in hand_box]  # 转换为整数
                    })
                    # 在图像上绘制检测框
                    cv2.rectangle(img, (int(hand_box[0]), int(hand_box[1])), (int(hand_box[2]), int(hand_box[3])), (0, 255, 0), 1)
                    cv2.putText(img, label, (int(hand_box[0]), int(hand_box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    results.append({
                        'label': label,
                        'xyxy': [int(coord) for coord in xyxy]  # 转换为整数
                    })
                    # 在图像上绘制检测框
                    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 1)
                    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 将图像转换回 base64 编码
    _, buffer = cv2.imencode('.jpg', img)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # 返回检测结果和带有框的图像
    return jsonify({'results': results, 'image': 'data:image/jpeg;base64,' + encoded_image})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
