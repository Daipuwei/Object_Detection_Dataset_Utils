# -*- coding: utf-8 -*-
# @Time    : 2021/7/31 下午4:55
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : cityscapes2foggy_cityscapes.py
# @Software: PyCharm

"""
    这是根据Cityscapes数据集和已有的Foggy Cityscapes图像集生成Foggy Cityscapes
    数据集的工具脚本，默认Cityscapes数据集已经转化为VOC数据集格式。 根据自身需要修改beta、
    cityscapes_dataset_dir、foggy_cityscapes_dataset_dir和foggy_image_dir即可
    完成Foggy Cityscapes数据集的生成。

    其中:
        cityscapes_dataset_dir代表已经转化为VOC数据集格式的cityscapes数据集目录地址，
        foggy_cityscapes_dataset_dir代表VOC数据集格式的foggy cityscapes数据集目录地址，
        foggy_image_dir为人工生成的带雾cityscapes图像数据目录地址，
        beta代表雾浓度。
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count


def cityscapes2foggy_cityscapes(cityscapes_dataset_dir,foggy_cityscapes_dataset_dir,foggy_image_dir,beta=0.01):
    """
    这是将cityscapes数据集转化为foggy_cityscapes数据集的函数，假定Cityscapes数据集已经转化成VOC数据集格式
    :param cityscapes_dataset_dir: cityscapes数据集目录
    :param foggy_cityscapes_dataset_dir: foggy_cityscapes数据集目录
    :param foggy_image_dir: 雾深度图像数据集目录
    :param beta: beta参数，用于控制雾的浓度，默认为0.01，候选值有[0.005,0.01,0.02]
    :return:
    """
    # 初始化Cityscapes和Foggy_Cityscape两个数据集相关路径
    foggy_cityscapes_image_dir = os.path.join(foggy_cityscapes_dataset_dir,"JPEGImages")
    foggy_cityscapes_label_dir = os.path.join(foggy_cityscapes_dataset_dir,"Annotations")
    foggy_cityscapes_main_dir = os.path.join(foggy_cityscapes_dataset_dir,"ImageSets","Main")
    if not os.path.exists(foggy_cityscapes_image_dir):
        os.makedirs(foggy_cityscapes_image_dir)
    if not os.path.exists(foggy_cityscapes_label_dir):
        os.makedirs(foggy_cityscapes_label_dir)
    if not os.path.exists(foggy_cityscapes_main_dir):
        os.makedirs(foggy_cityscapes_main_dir)

    # 初始化Foggy_CItyscapes数据集的XML文件路径
    cityscapes_label_dir = os.path.join(cityscapes_dataset_dir,"Annotations")
    cityscapes_label_paths = []
    foggy_cityscapes_label_paths = []
    for annotation_name in os.listdir(cityscapes_label_dir):
        name,ext = os.path.splitext(annotation_name)
        cityscapes_label_paths.append(os.path.join(cityscapes_label_dir,annotation_name))
        foggy_cityscapes_label_paths.append(os.path.join(foggy_cityscapes_label_dir,name+"_foggy_beta_{}".format(beta)+ext))

    # 初始化Foggy_Cityscapes数据集图像路径
    foggy_image_paths = []
    foggy_cityscapes_image_paths = []
    for choice in ["train","val"]:
        # 初始化子数据集目录路径
        _foggy_cityscapes_image_dir = os.path.join(foggy_image_dir, choice)
        with open(os.path.join(foggy_cityscapes_main_dir,choice+".txt"),'w') as f:
            for city_name in os.listdir(_foggy_cityscapes_image_dir):
                city_dir = os.path.join(_foggy_cityscapes_image_dir,city_name)
                for image_name in os.listdir(city_dir):
                    if "beta_{}".format(beta) in image_name:
                        name,ext = os.path.splitext(image_name)
                        if is_contain_object(name,foggy_cityscapes_label_paths):            # Foggy Cityscapes图像存在目标
                            foggy_image_paths.append(os.path.join(city_dir,image_name))
                            foggy_cityscapes_image_paths.append(os.path.join(foggy_cityscapes_image_dir,image_name))
                            f.write(name+"\n")         # 写入文件名称到指定txt文件
    # 将数据集图像名称写入voc数据集的trainval.txt文件
    with open(os.path.join(foggy_cityscapes_main_dir, 'trainval.txt'), "w+") as f:
        for cityscapes_image_path in foggy_cityscapes_image_paths:
            dir, image_name = os.path.split(cityscapes_image_path)
            name, ext = os.path.splitext(image_name)
            f.write(name + "\n")

    # 多线程将Foggy_Cityscapes数据集图像复制到Cityscapes数据集目录里，假定Cityscapes数据集已经转化成VOC数据集格式
    size = len(cityscapes_label_paths)
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for i,start in enumerate(np.arange(0,size,batch_size)):
        end = int(np.min([start+batch_size,size]))
        batch_cityscapes_label_paths = cityscapes_label_paths[start:end]
        batch_foggy_cityscapes_label_paths = foggy_cityscapes_label_paths[start:end]
        batch_foggy_image_paths = foggy_image_paths[start:end]
        batch_foggy_cityscapes_image_paths = foggy_cityscapes_image_paths[start:end]
        print("线程{}处理{}张图像".format(i,len(batch_foggy_cityscapes_image_paths)))
        pool.apply_async(batch_copy_image_label,error_callback=print_error,
                         args=(batch_foggy_image_paths,batch_foggy_cityscapes_image_paths,
                               batch_cityscapes_label_paths,batch_foggy_cityscapes_label_paths))
    pool.close()
    pool.join()

def batch_copy_image_label(batch_foggy_image_paths,batch_foggy_cityscapes_image_paths,
                           batch_cityscapes_label_paths,batch_foggy_cityscapes_label_paths):
    """
    这是批量复制图像函数，将Foggy_Cityscapes数据集图像复制到Cityscapes数据集里
    :param batch_foggy_image_paths: 批量foggy图像路径数组
    :param batch_foggy_cityscapes_image_paths: 批量foggy_cityscapes数据集图像路径数组
    :param batch_cityscapes_label_paths: 批量cityscapes数据集标签路径数组
    :param batch_foggy_cityscapes_label_paths: 批量foggy_cityscape数据集标签路径数组
    :return:
    """
    size = len(batch_foggy_image_paths)
    for i in tqdm(np.arange(size)):
        foggy_image_path = batch_foggy_image_paths[i]
        foggy_cityscapes_image_path = batch_foggy_cityscapes_image_paths[i]
        cityscapes_label_path = batch_cityscapes_label_paths[i]
        foggy_cityscapes_label_path = batch_foggy_cityscapes_label_paths[i]
        copy_image_label(foggy_image_path,foggy_cityscapes_image_path,
                         cityscapes_label_path,foggy_cityscapes_label_path)

def copy_image_label(foggy_image_path,foggy_cityscapes_image_path,
                     cityscapes_label_path,foggy_cityscapes_label_path):
    """
    这是复制图像及其XML标签的函数
    :param foggy_image_path: foggy图像路径
    :param foggy_cityscapes_image_path: foggy_cityscapes数据集图像路径
    :param cityscapes_label_path: cityscapes数据集标签路径
    :param foggy_cityscapes_label_path: foggy_cityscape数据集标签路径
    :return:
    """
    # 复制图像
    image = cv2.imread(foggy_image_path)
    cv2.imwrite(foggy_cityscapes_image_path,image)

    # 复制XML文件
    in_file = open(cityscapes_label_path)
    tree = ET.parse(in_file)
    tree.write(foggy_cityscapes_label_path)

def print_error(value):
    """
    定义自己的回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def is_contain_object(image_name,foggy_cityscapes_label_paths):
    """
    这是Foggy Cityscapes数据集中一张图像是否存在目标的函数
    :param image_name: 图像名称
    :param foggy_cityscapes_label_paths: Foggy Cityscapes数据集图像路径数组
    :return:
    """
    flag = False
    for foggy_image_path in foggy_cityscapes_label_paths:
        if image_name in foggy_image_path:
            flag = True
            break
    return flag

def run_main():
    """
    这是主函数
    """
    # Cityscapes --> Foggy_Cityscapes_beta_0.005
    print("Cityscapes --> Foggy_Cityscapes_beta=0.005 Start")
    beta = 0.005
    cityscapes_dataset_dir = os.path.abspath("../dataset/Cityscapes")
    foggy_cityscapes_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_{}".format(beta))
    foggy_image_dir = os.path.abspath("../object_detection_dataset/cityscapes_foggy/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy")
    cityscapes2foggy_cityscapes(cityscapes_dataset_dir,foggy_cityscapes_dataset_dir,foggy_image_dir,beta)
    print("Cityscapes --> Foggy_Cityscapes_beta_0.005 Finish")

    # Cityscapes --> Foggy_Cityscapes_beta=0.01
    print("Cityscapes --> Foggy_Cityscapes_beta_0.01 Start")
    beta = 0.01
    cityscapes_dataset_dir = os.path.abspath("../dataset/Cityscapes")
    foggy_cityscapes_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_{}".format(beta))
    foggy_image_dir = os.path.abspath("../object_detection_dataset/cityscapes_foggy/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy")
    cityscapes2foggy_cityscapes(cityscapes_dataset_dir,foggy_cityscapes_dataset_dir,foggy_image_dir,beta)
    print("Cityscapes --> Foggy_Cityscapes_beta_0.01 Finish")

    # Cityscapes --> Foggy_Cityscapes_beta=0.02
    print("Cityscapes --> Foggy_Cityscapes_beta_0.02 Start")
    beta = 0.02
    cityscapes_dataset_dir = os.path.abspath("../dataset/Cityscapes")
    foggy_cityscapes_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_{}".format(beta))
    foggy_image_dir = os.path.abspath("../object_detection_dataset/cityscapes_foggy/leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy")
    cityscapes2foggy_cityscapes(cityscapes_dataset_dir,foggy_cityscapes_dataset_dir,foggy_image_dir,beta)
    print("Cityscapes --> Foggy_Cityscapes_beta_0.02 Finish")

if __name__ == '__main__':
    run_main()