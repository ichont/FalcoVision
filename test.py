import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 定义数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载类别名称
data_dir = 'test'
class_names = sorted(os.listdir(data_dir))  # 确保排序后的类别名称
num_classes = len(class_names)
print(class_names)
print(f"类别数量：{num_classes}")

# 加载模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('savedmodel.pth'))
model.eval()  # 设置为评估模式

# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict_image(image_path):
    """对单张图片进行预测"""
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image)
    image = image.unsqueeze(0)  # 增加一个批次维度
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    return preds[0]

# 计算准确率
path = 'test'
total = 0
correct = 0
predicted_labels = []
true_labels = []

for root, _, files in os.walk(path):
    for file in files:
        total += 1
        image_path = os.path.join(root, file)
        predicted_class_idx = predict_image(image_path)
        predicted_class = class_names[predicted_class_idx]
        true_class = os.path.split(root)[-1]

        predicted_labels.append(predicted_class)
        true_labels.append(true_class)

        if predicted_class == true_class:
            correct += 1
        else:
            print(f'预测错误: {file} 预测类别为 {predicted_class}, 实际类别为 {true_class}')

accuracy = correct / total * 100
print(f'总共{total}张图片，预测正确{correct}张图片，准确率为{accuracy:.2f}%')