# TODO: 训练模型

from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
# load
model = YOLO('yolov8n.pt')

trained_model = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=320,
    batch=64,
    device='cpu',
    project='./runs',
    name='detect_train'
)
print("It has already trained.")

# === 替换后的分析模块 ===


best_model = YOLO('./runs/detect_train/weights/best.pt')

# 2. 使用新模型进行验证
metrics = best_model.val(
    data="data.yaml",
    split='val',
    device='cpu'
)
print("验证集性能报告:")
print(f"1. mAP50: {metrics.box.map50:.3f}")          # 目标检测常用指标
print(f"2. mAP50-95: {metrics.box.map:.3f}")        # 更严格的mAP指标
print(f"3. 精确率: {metrics.box.p[0]:.3f}")              # 取第一个类别的精确率
print(f"4. 召回率: {metrics.box.r[0]:.3f}\n")            # 取第一个类别的召回率

# 绘制训练日志中的损失曲线（从CSV文件读取）
import pandas as pd
log_csv = pd.read_csv('runs/detect_train/results.csv')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.figure(figsize=(12, 6))
plt.plot(log_csv['train/box_loss'], label='训练损失')
plt.plot(log_csv['val/box_loss'], label='验证损失')
plt.title('损失曲线')
plt.legend()
plt.savefig('training_metrics.png')


