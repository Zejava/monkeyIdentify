import sys
from pathlib import Path
from ultralytics import YOLO
from openvino.runtime import Core, serialize



def convert_model():
    """执行模型转换全流程"""
    # 模型路径配置
    model_dir = Path('runs/detect_train/weights')
    pt_path = model_dir / 'best.pt'

    # 加载PyTorch模型
    model = YOLO(str(pt_path))

    # 导出ONNX格式
    print("正在导出ONNX模型...")
    model.export(
        format='onnx',
        imgsz=320,  # 保持与训练时相同的输入尺寸
        simplify=True,  # 启用模型简化
        opset=12,  # 指定ONNX算子版本
        dynamic=False  # 禁用动态尺寸
    )

    # OpenVINO转换
    print("正在转换OpenVINO IR格式...")
    ie = Core()
    onnx_path = pt_path.with_suffix('.onnx')

    try:
        ov_model = ie.read_model(onnx_path)
        serialize(ov_model, str(model_dir / 'best.xml'))
        print("转换成功！")
    except Exception as e:
        print(f"转换失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    convert_model()