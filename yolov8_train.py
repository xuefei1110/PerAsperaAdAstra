from ultralytics import YOLO
##########
import torch
print(torch.cuda.is_available())  # 应该返回 True
print(torch.cuda.get_device_name(0))  # 显示你的显卡型号

##########
def train_yolov8():
    # 创建 YOLOv8 模型（这里以 YOLOv8n 为例，你可以改为 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x' 等）
    model = YOLO('yolov8n.pt')

    # 开始训练
    model.train(
        data='D:\\yolov8\\ttpla.yaml',       # 数据集配置文件
        epochs=100,               # 训练轮数
        imgsz=1024,                # 输入图像大小
        batch=16,                 # 批大小
        name='yolov8_train_exp',  # 保存结果的子目录名
        project='runs/train',     # 保存路径的根目录
        device=0 ,                  # 使用 GPU:0，如果是 CPU 就改为 'cpu'
        workers = 0,
    )


if __name__ == '__main__':
    train_yolov8()
