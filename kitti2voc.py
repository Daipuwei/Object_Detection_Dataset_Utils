# -*- coding: utf-8 -*-
# @Time    : 2021/8/7 下午1:01
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : kitti2voc.py
# @Software: PyCharm

"""
    这是KITTI数据集转化为VOC数据集格式的脚本，根据开发需要自行修改
    kitti_dataset_dir、voc_dataset_dir、train_ratio、class_names即可。

    其中：
        - kitti_dataset_dir为原始KITTI数据集目录路径；
        - voc_dataset_dir为VOC数据集格式的KITTI数据集目录路径；
        - train_ratio为训练集比例，默认为0.8；
        - class_names为目标名称数组，默认为['Person_sitting',"Pedestrian",'Cyclist',"Truck","Car","Tram","Van"]；
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

def kitti2voc(kitti_dataset_dir,voc_dataset_dir,train_ratio=0.8,
              class_names=['Person_sitting',"Pedestrian",'Cyclist',"Truck","Car","Tram","Van"]):
    """
    这是将KITTI数据集转化为VOC数据集的函数
    :param kitti_dataset_dir: kitti数据集目录路径
    :param voc_dataset_dir: voc数据集目录路径
    :param train_ratio: 训练集比例，默认为0.8
    :param class_names: 目标名称数组，默认为[]
    :return:
    """
    # 初始化kitti数据集相关路径
    kitti_image_dir = os.path.join(kitti_dataset_dir,'training','image_2')
    kitti_label_dir = os.path.join(kitti_dataset_dir,'training','label_2')
    #print(kitti_image_dir,kitti_label_dir)

    # 初始化voc数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    voc_imagesets_main_dir = os.path.join(voc_dataset_dir, "ImageSets", "Main")
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    if not os.path.exists(voc_annotation_dir):
        os.makedirs(voc_annotation_dir)
    if not os.path.exists(voc_imagesets_main_dir):
        os.makedirs(voc_imagesets_main_dir)

    # 初始化图像和标签文件路径
    kitti_image_paths = []
    kitti_label_paths = []
    voc_image_paths = []
    voc_annotation_paths = []
    for txt_name in os.listdir(kitti_label_dir):
        txt_label_path = os.path.join(kitti_label_dir,txt_name)
        #print(txt_label_path)
        #print(is_conitain_object(txt_label_path,class_names))
        if is_conitain_object(txt_label_path,class_names):
            name,ext = os.path.splitext(txt_name)
            kitti_image_paths.append(os.path.join(kitti_image_dir,name+".png"))
            kitti_label_paths.append(os.path.join(kitti_label_dir,txt_name))
            voc_image_paths.append(os.path.join(voc_image_dir,name+".jpg"))
            voc_annotation_paths.append(os.path.join(voc_annotation_dir,name+".xml"))
    kitti_image_paths = np.array(kitti_image_paths)
    kitti_label_paths = np.array(kitti_label_paths)
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    print(len(kitti_image_paths),len(kitti_label_paths),len(voc_image_paths),len(voc_annotation_paths))

    # 随机打乱数据集
    size = len(kitti_label_paths)
    random_index = np.random.permutation(size)
    kitti_image_paths = kitti_image_paths[random_index]
    kitti_label_paths = kitti_label_paths[random_index]
    voc_image_paths = voc_image_paths[random_index]
    voc_annotation_paths = voc_annotation_paths[random_index]

    # 随机划分训练集和测试集
    train_size = int(size*train_ratio)
    train_kitti_image_paths = kitti_image_paths[0:train_size]
    val_kitti_image_paths = kitti_image_paths[train_size:]
    for choice,kitti_image_paths in [('train',train_kitti_image_paths),
                               ('val',val_kitti_image_paths),
                               ('trainval',kitti_image_paths)]:
        voc_txt_path = os.path.join(voc_imagesets_main_dir,choice+".txt")
        with open(voc_txt_path,'w') as f:
            for image_path in kitti_image_paths:
                dir,image_name = os.path.split(image_path)
                name,ext = os.path.splitext(image_name)
                f.write(name+"\n")
    # 利用多线程将KITTI数据集转换为VOC数据集格式
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_kitti_image_paths = kitti_image_paths[start:end]
        batch_kitti_label_paths = kitti_label_paths[start:end]
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        pool.apply_async(batch_image_label_process,error_callback=print_error,
                         args=(batch_kitti_image_paths,batch_kitti_label_paths,
                               batch_voc_image_paths,batch_voc_annotation_paths,class_names))
    pool.close()
    pool.join()

def is_conitain_object(kitti_txt_path,class_names):
    """
    这是判断kitti数据集的txt标签中是否包含候选目标实例的函数
    :param kitti_txt_path: kitti数据集的json标签文件路径
    :param class_names: 目标分类数组
    :return:
    """
    flag = False
    with open(kitti_txt_path,'r') as f:
        for line in f.readlines():
            strs = line.strip().split(" ")
            if strs[0] not in class_names:
                continue
            else:
                flag = True
                break
    return flag

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def single_image_label_process(kitti_image_path,kitti_txt_path,voc_image_path,voc_annotation_path,class_names):
    """
    这是将单张kitti图像及其txt标签转换为VOC数据集格式的函数
    :param kitti_image_path: kitti图像路径
    :param kitti_txt_path: kitti标签路径
    :param voc_image_path: voc图像路径
    :param voc_annotation_path: voc标签路径
    :param class_names: 目标名称数组
    :return:
    """
    # 初始化VOC标签写类,复制图像
    image = cv2.imread(kitti_image_path)
    cv2.imwrite(voc_image_path,image)
    h, w, c = np.shape(image)
    dir,image_name = os.path.split(kitti_image_path)
    writer = Writer(image_name,w,h)

    # TXT标签转化为VOC标签
    with open(kitti_txt_path,'r') as f:                 # 加载txt标签
        for line in f.readlines():
            strs = line.strip().split(" ")
            if strs[0] not in class_names:
                continue
            else:
                xmin = max(int(float(strs[4]))-1,0)
                ymin = max(int(float(strs[5]))-1,0)
                xmax = min(int(float(strs[6]))+1,w)
                ymax = min(int(float(strs[7]))+1,h)
                label = strs[0]
                if label in ['Person_sitting',"Pedestrian"]:
                    label = 'person'
                elif label in ['Cyclist']:
                    label = 'rider'
                elif label in ['Car','Van']:
                    label = 'car'
                elif label in ['Truck']:
                    label = 'truck'
                elif label in ['Tram']:
                    label = 'bus'
                writer.addObject(label,xmin,ymin,xmax,ymax)
        writer.save(voc_annotation_path)

def batch_image_label_process(batch_kitti_image_paths,batch_kitti_txt_paths,
                              batch_voc_image_paths,batch_voc_annotation_paths,class_names):
    """
    批量处理kitti数据转化为voc数据
    :param batch_kitti_image_paths: 批量kitti图像路径数组
    :param batch_kitti_txt_paths: 批量kitti标签路径数组
    :param batch_voc_image_paths: 批量voc图像路径数组
    :param batch_voc_annotation_paths: 批量voc标签路径数组
    :param class_names: 目标名称数组
    :return:
    """
    size = len(batch_kitti_image_paths)
    for i in tqdm(np.arange(size)):
        kitti_image_path = batch_kitti_image_paths[i]
        kitti_txt_path = batch_kitti_txt_paths[i]
        voc_image_path = batch_voc_image_paths[i]
        voc_annotation_path = batch_voc_annotation_paths[i]
        single_image_label_process(kitti_image_path,kitti_txt_path,
                                   voc_image_path,voc_annotation_path,class_names)

def run_main():
    """
    这是主函数
    """
    # KITTI --> VOC
    print("KITTI --> VOC Start")
    kitti_dataset_dir = os.path.abspath("../KITTI")
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/KITTI")
    train_ratio = 0.8
    class_names=['Person_sitting',"Pedestrian",'Cyclist',"Truck","Car","Tram","Van"]
    kitti2voc(kitti_dataset_dir,voc_dataset_dir,train_ratio,class_names)
    print("KITTI --> VOC Start")

if __name__ == '__main__':
    run_main()
