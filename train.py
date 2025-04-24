
import os
import argparse
import sys
import torch.nn as nn
import torch
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import json
import torch.utils.data as Data
from tqdm import tqdm

from model import *
import torch.optim as optim


def get_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=10,help='number of epochs')
    parser.add_argument('--batch_size',type=int,default=8,help='batch size')
    parser.add_argument('--data_path',type=str,default='./dataset/',help='path to dataset')
    parser.add_argument('--model',type=str,default='resnet18',help='model name')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--save_dir',type=str,default='./checkpoint/',help='save .pth')
    parser.add_argument('--num_classes',type=int,default=5,help='number of classes')


    return parser


def train(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Use device:', device)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),    # 图像缩放
        transforms.CenterCrop(224),    # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'train'), transform=train_transform)
    train_num = len(train_dataset)
    val_dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'test'), transform=val_transform)
    val_num = len(val_dataset)

    flower_list = train_dataset.class_to_idx
    class_dict = dict((val,key) for key,val in flower_list.items())

    json_str = json.dumps(class_dict,indent=4)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print("Using batch_size={} dataloader worker every process.".format(num_workers))

    train_loader = Data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=num_workers)
    val_loader = Data.DataLoader(val_dataset,batch_size=args.batch_size,num_workers=num_workers,shuffle=False)
    print('Number of training images:{}, Number of validation images:{}'.format(train_num,val_num))

    model = get_model(args.model)
    num_ftrs = model.fc.in_features  # 获取全连接层的输入特征数量
    model.fc = torch.nn.Linear(num_ftrs, len(flower_list))  # 修改输出维度为5
    model = model.cuda()

    loss_function = nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params,args.lr)

    batch_num = len(train_loader)
    total_time = 0
    best_acc = 0

    for epoch in range(args.epochs):

        start_time = time.perf_counter()

        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader,file=sys.stdout)

        for step,data in enumerate(train_bar):
            train_images,train_labels = data

            train_images = train_images.to(device)
            train_labels = train_labels.to(device)

            optimizer.zero_grad()
            outputs = model(train_images)
            loss = loss_function(outputs,train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            train_bar.desc = "train eopch[{}/{}] loss: {:.3f}".format(epoch+1,args.epochs,loss)

        model.eval()
        val_acc = 0
        var_bar = tqdm(val_loader,file=sys.stdout)

        with torch.no_grad():
            for val_data in var_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_y=model(val_images)
                pred_y = torch.max(val_y,1)[1]

                val_acc += torch.eq(pred_y,val_labels).sum().item()

                var_bar.desc = "val eopch[{}/{}]".format(epoch+1,args.epochs)

        val_accurate = val_acc / val_num
        print("[epoch {:.0f}] train_loss: {:.3f} val_accuracy: {:.3f}".format(epoch+1,train_loss/batch_num,val_accurate))

        epoch_time = time.perf_counter()-start_time
        print("epoch_time:{}".format(epoch_time))
        total_time += epoch_time
        print()

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(),os.path.join(args.save_dir,args.model+'_best.pth'))

    m,s = divmod(total_time,60)
    h,m = divmod(m,60)
    print("total time:{:0f}:{:0f}:{:0f}".format(h,m,s))
    print("Finished Training!")




if __name__ == '__main__':
    args = get_argparse().parse_args()
    train(args)
