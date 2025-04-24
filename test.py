import argparse
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from model import *
import torchvision.transforms as transforms
import cv2
import os
import json

def get_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='./dataset/show',help='path to dataset')
    parser.add_argument('--model',type=str,default='resnet18',help='model name')
    parser.add_argument('--checkpoint',type=str,default='./checkpoint/resnet18_best.pth',help='checkpoint path')
    parser.add_argument('--num_classes',type=int,default=5,help='number of classes')

    return parser

def main():
    args = get_argparse().parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    image_path = [os.path.join(args.data_path,f) for f in os.listdir(args.data_path) if f.endswith('.jpg')]
    image_path = image_path[:10]

    class_indict = json.load(open("./class_indices.json"))

    model = get_model(args.model)
    num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数量
    model.fc = torch.nn.Linear(num_ftrs, args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    model.eval()
    fig , axes = plt.subplots(2,5,figsize=(15,6))
    axes = axes.flatten()

    for idx, image_path in enumerate(image_path):

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = data_transform(img).unsqueeze(0)
        img = img.to(device)

        with torch.no_grad():
            output = model(img)
            predict = torch.softmax(output, dim=1)
            pred = torch.argmax(predict,dim=1).cpu().numpy()

        # print_res = "class: {} prob: {:.3}".format(class_indict[str(pred)],
        #                                            predict[pred].cpu().numpy())
        class_name = class_indict[str(pred[0])]  # 获取类别名称
        prob = predict[0, pred[0]].item()  # 获取预测概率
        print_res = "class: {} prob: {:.3f}".format(class_name, prob)

        # 显示图片和标题
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        img_for_display = img.squeeze().mul(std).add(mean).permute(1, 2, 0).cpu().numpy() # 调整通道顺序并移动到 CPU
        axes[idx].imshow(img_for_display)  # 显示图像
        axes[idx].set_title(print_res)
        axes[idx].axis('off')  # 不显示坐标轴

    # 隐藏多余的子图（如果有）
    for idx in range(len(image_path), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()


