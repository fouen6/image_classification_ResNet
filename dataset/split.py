import os
import glob
import random
import cv2
import numpy as np

if __name__ == '__main__':
    split_rate = 0.2  # 训练集和验证集划分比率
    resize_image = 224  # 图片缩放后统一大小
    file_path = './flower_photos'  # 获取原始数据集路径

    # 找到文件中所有文件夹的目录，即类文件夹名
    dirs = glob.glob(os.path.join(file_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    print("Totally {} classes: {}".format(len(dirs), dirs))  # 打印花类文件夹名称

    for path in dirs:
        # 对每个类别进行单独处理
        path = path.split('\\')[-1]  # -1表示以分隔符/保留后面的一段字符

        # 在根目录中创建两个文件夹，train/test
        os.makedirs("train\\{}".format(path), exist_ok=True)
        os.makedirs("test\\{}".format(path), exist_ok=True)

        # 读取原始数据集中path类中对应类型的图片，并添加到files中
        files = glob.glob(os.path.join(file_path, path, '*jpg'))
        files += glob.glob(os.path.join(file_path, path, '*jpeg'))
        files += glob.glob(os.path.join(file_path, path, '*png'))

        random.shuffle(files)  # 打乱图片顺序
        split_boundary = int(len(files) * split_rate)  # 训练集和测试集的划分边界

        for i, file in enumerate(files):
            img = cv2.imread(file)

            # 更改原始图片尺寸
            old_size = img.shape[:2]  # (height, width)
            ratio = float(resize_image) / max(old_size)  # 通过最长的size计算原始图片缩放比率
            # 把原始图片最长的size缩放到resize_pic，短的边等比率缩放，等比例缩放不会改变图片的原始长宽比
            new_size = tuple([int(x * ratio) for x in old_size])

            im = cv2.resize(img, (new_size[1], new_size[0]))  # 更改原始图片的尺寸
            new_im = np.zeros((resize_image, resize_image, 3), dtype=np.uint8)  # 创建一个resize_pic尺寸的黑色背景
            # 把新图片im贴到黑色背景上，并通过'地板除//'设置居中放置
            x_start = (resize_image - new_size[1]) // 2
            y_start = (resize_image - new_size[0]) // 2
            new_im[y_start:y_start + new_size[0], x_start:x_start + new_size[1]] = im

            # 打印处理进度
            print("Processing file {} of {}: {}".format(i + 1, len(files), file))

            # 先划分0.2_rate的测试集，剩下的再划分为0.9_ate的训练集，同时直接更改图片后缀为.jpg
            if i < split_boundary:
                cv2.imwrite(os.path.join("test\\{}".format(path),
                                         file.split('\\')[-1].split('.')[0] + '.jpg'), new_im)
            else:
                cv2.imwrite(os.path.join("train\\{}".format(path),
                                         file.split('\\')[-1].split('.')[0] + '.jpg'), new_im)

    # 统计划分好的训练集和测试集中.jpg图片的数量
    train_files = glob.glob(os.path.join('train', '*', '*.jpg'))
    test_files = glob.glob(os.path.join('test', '*', '*.jpg'))

    print("Totally {} files for train".format(len(train_files)))
    print("Totally {} files for test".format(len(test_files)))
