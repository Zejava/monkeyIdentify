from ultralytics import YOLO
import os
#  自动标注脚本
# 定义函数，用于生成YOLO格式的标注文件
# 初始化预训练模型
#使用yolov8n.pt 是最轻量的模型，推理速度最快，适合快速标注
model = YOLO('yolov8n.pt')  # 使用官方预训练模型



def generate_labels(base_image_dir, base_label_dir):
    os.makedirs(base_label_dir, exist_ok=True)
    for class_dir in os.listdir(base_image_dir):
        class_path = os.path.join(base_image_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        label_class_dir = os.path.join(base_label_dir, class_dir)
        os.makedirs(label_class_dir, exist_ok=True)
        # 处理当前类别的图片
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            # 执行目标检测
            results = model.predict(img_path, conf=0.5)#conf=0.5是只保留置信度大于 0.5 的检测框。
            # 生成标注文件路径时使用新的目录结构
            label_path = os.path.join(label_class_dir,
                                      os.path.splitext(img_name)[0] + '.txt')
            with open(label_path, 'w') as f:
                for box in results[0].boxes:
                    # 转换坐标为YOLO格式 (中心点坐标+宽高，归一化值)
                    x_center = (box.xywh[0][0].item() / results[0].orig_shape[1])
                    y_center = (box.xywh[0][1].item() / results[0].orig_shape[0])
                    width = (box.xywh[0][2].item() / results[0].orig_shape[1])
                    height = (box.xywh[0][3].item() / results[0].orig_shape[0])
            class_id = int(class_dir[1:])  # 提取n后面的数字
            with open(label_path, 'w') as f:
                for box in results[0].boxes:
                    f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')

# 修改调用方式（去除原路径中的images目录）
#训练集
generate_labels(
    r'C:\Users\pzq1\Downloads\monkey\training\training',
    r'C:\Users\pzq1\Downloads\monkey\training\labels'
)
#验证集
generate_labels(
    r'C:\Users\pzq1\Downloads\monkey\validation\validation',
    r'C:\Users\pzq1\Downloads\monkey\validation\labels'
)
