from ultralytics import YOLO

def main():
    # 加载训练好的模型
    model = YOLO('runs/train/yolov8_train_exp6/weights/best.pt')  # 替换成你的模型路径

    # 进行验证
    results = model.val(data='D:\\yolov8\\ttpla.yaml')  # 替换成你的数据集配置文件路径

    # 打印验证结果
    print(results)

if __name__ == '__main__':
    main()
