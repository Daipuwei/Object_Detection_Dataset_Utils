# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 11:07
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : random_split_voc_dataset.py
# @Software: PyCharm

"""
    这是随机划分VOC数据集的脚本
"""

import os
import numpy as np

def random_split_voc_dataset(voc_dataset_dir,train_ratio=0.8):
    """
    这是随机划分VOC数据集的函数
    Args:
        voc_dataset_dir: voc数据集地址
        train_ratio: 训练集比例，默认为0.8
    Returns:
    """
    # 初始化相关文件和文件夹路径
    voc_main_dir = os.path.join(voc_dataset_dir,'ImageSets','Main')
    voc_image_dir = os.path.join(voc_dataset_dir,'JPEGImages')
    train_txt_path = os.path.join(voc_main_dir,'train.txt')
    trainval_txt_path = os.path.join(voc_main_dir,'trainval.txt')
    val_txt_path = os.path.join(voc_main_dir,'val.txt')
    if not os.path.exists(voc_main_dir):
        os.makedirs(voc_main_dir)

    # 遍历图像文件夹，获取所有图像
    image_name_list = []
    for image_name in os.listdir(voc_image_dir):
        image_name_list.append(image_name)
    image_name_list = np.array(image_name_list)
    image_name_list = np.random.permutation(image_name_list)

    # 划分训练集和测试集
    size = len(image_name_list)
    random_index = np.random.permutation(size)
    train_size = int(size*train_ratio)
    train_image_name_list = image_name_list[random_index[0:train_size]]
    val_image_name_list = image_name_list[random_index[train_size:]]

    # 生成trainval
    with open(trainval_txt_path,'w') as f:
        for image_name in image_name_list:
            fname,ext = os.path.splitext(image_name)
            f.write(fname+"\n")
    # 生成train
    with open(train_txt_path, 'w') as f:
        for image_name in train_image_name_list:
            fname, ext = os.path.splitext(image_name)
            f.write(fname + "\n")
    # 生成val
    with open(val_txt_path,'w') as f:
        for image_name in val_image_name_list:
            fname, ext = os.path.splitext(image_name)
            f.write(fname + "\n")

def run_main():
    """
    这是主函数
    """
    train_ratio = 0.8
    voc_dataset_dir = os.path.abspath("../dataset/Cityscapes"),
    random_split_voc_dataset(voc_dataset_dir, train_ratio)

if __name__ == '__main__':
    run_main()
