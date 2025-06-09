import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
)
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from openvino.runtime import Core

## ui界面
class ImageRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenVINO 图像识别")
        self.image_path = None
        self.core = Core()

        # 加载模型（只加载一次）
        self.model = self.core.read_model(model="runs/detect_train/weights/best.xml")
        self.compiled_model = self.core.compile_model(model=self.model, device_name="CPU")
        self.original_model = YOLO("runs/detect_train/weights/best.pt")
        self.infer_request = self.compiled_model.create_infer_request()

        # 创建 UI
        self.layout = QVBoxLayout()

        self.image_label = QLabel("未选择图片")
        self.image_label.setFixedSize(320, 320)
        self.image_label.setStyleSheet("border: 1px solid black")
        self.layout.addWidget(self.image_label)

        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.upload_image)
        self.layout.addWidget(self.upload_button)

        self.recognize_button = QPushButton("识别图片")
        self.recognize_button.clicked.connect(self.recognize_image)
        self.layout.addWidget(self.recognize_button)

        self.result_label = QLabel("识别结果将在此显示")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path).scaled(320, 320)
            self.image_label.setPixmap(pixmap)

    def recognize_image(self):
        if not self.image_path:
            self.result_label.setText("请先上传图片")
            return

        # 读取图片并进行推理
        image = cv2.imread(self.image_path)
        results = self.original_model(image)  # YOLO 推理

        # 将预测结果绘制到图片上
        result_img = results[0].plot()  # 画出检测框和标签（BGR格式）

        # 转换为 QPixmap 以便显示在 QLabel 中
        rgb_image = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(320, 320)

        self.image_label.setPixmap(pixmap)

        # 显示检测到的类别名
        names = [
            'mantled_howler',
            'patas_monkey',
            'bald_uakari',
            'japanese_macaque',
            'pygmy_marmoset',
            'white_headed_capuchin',
            'silvery_marmoset',
            'common_squirrel_monkey',
            'black_headed_night_monkey',
            'nilgiri_langur'
        ]

        if results[0].boxes.cls.numel() > 0:
            class_ids = results[0].boxes.cls.int().tolist()
            detected = [names[cid] for cid in class_ids if 0 <= cid < len(names)]
            self.result_label.setText("识别结果：" + "，".join(detected))
        else:
            self.result_label.setText("未检测到目标")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageRecognitionApp()
    window.show()
    sys.exit(app.exec_())
