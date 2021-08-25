# -*- coding: utf-8 -*-
# @Time    : 2021/8/5 下午1:52
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : copy_voc_dataset.py
# @Software: PyCharm

"""
    这是根据指定目标类别复制VOC数据集的函数
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

def copy_voc_dataset(voc_dataset_dir,new_voc_dataset_dir,class_names,train_ratio=0.8):
    """
    这是根据指定目标类别复制VOC数据集的函数
    :param voc_dataset_dir: voc数据集地址
    :param new_voc_dataset_dir: 新voc数据集地址
    :param class_names: 目标类别数组
    :param train_ratio: 训练集比例
    :return:
    """
    # 初始化voc数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    new_voc_image_dir = os.path.join(new_voc_dataset_dir, 'JPEGImages')
    new_voc_annotation_dir = os.path.join(new_voc_dataset_dir, "Annotations")
    new_voc_imagesets_main_dir = os.path.join(new_voc_dataset_dir, "ImageSets", "Main")
    if not os.path.exists(new_voc_image_dir):
        os.makedirs(new_voc_image_dir)
    if not os.path.exists(new_voc_annotation_dir):
        os.makedirs(new_voc_annotation_dir)
    if not os.path.exists(new_voc_imagesets_main_dir):
        os.makedirs(new_voc_imagesets_main_dir)

    # 筛选包含指定目标的VOC数据集的图像和xml文件路径
    voc_image_paths = []
    voc_annotation_paths = []
    new_voc_image_paths = []
    new_voc_annotation_paths = []
    new_image_names = []
    for image_name in os.listdir(voc_image_dir):
        name,ext = os.path.splitext(image_name)
        image_path = os.path.join(voc_image_dir,image_name)
        xml_path = os.path.join(voc_annotation_dir,name+".xml")
        new_image_path = os.path.join(new_voc_image_dir, image_name)
        new_xml_path = os.path.join(new_voc_annotation_dir, name + ".xml")
        if is_contain_classes(xml_path,class_names):
            voc_image_paths.append(image_path)
            voc_annotation_paths.append(xml_path)
            new_voc_image_paths.append(new_image_path)
            new_voc_annotation_paths.append(new_xml_path)
            new_image_names.append(name)
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    new_voc_image_paths = np.array(new_voc_image_paths)
    new_voc_annotation_paths = np.array(new_voc_annotation_paths)
    new_image_names = np.array(new_image_names)

    # 随机打乱图像
    size = len(new_image_names)
    random_index = np.random.permutation(size)
    voc_image_paths = voc_image_paths[random_index]
    voc_annotation_paths = voc_annotation_paths[random_index]
    new_voc_image_paths = new_voc_image_paths[random_index]
    new_voc_annotation_paths = new_voc_annotation_paths[random_index]
    new_image_names = new_image_names[random_index]

    # 随机划分训练集和测试集
    train_size = int(size*train_ratio)
    train_new_image_names = new_image_names[0:train_size]
    val_new_image_names = new_image_names[train_size:]
    new_voc_trainval_txt_path = os.path.join(new_voc_imagesets_main_dir, 'trainval.txt')
    new_voc_train_txt_path = os.path.join(new_voc_imagesets_main_dir,'train.txt')
    new_voc_val_txt_path = os.path.join(new_voc_imagesets_main_dir,'val.txt')
    with open(new_voc_train_txt_path,'w') as f:
        for image_name in train_new_image_names:
            f.write(image_name+"\n")
    with open(new_voc_val_txt_path, 'w') as f:
        for image_name in val_new_image_names:
            f.write(image_name + "\n")
    with open(new_voc_trainval_txt_path, 'w') as f:
        for image_name in new_image_names:
            f.write(image_name + "\n")

    # 多线程处理复制图像和xml文件
    size = len(voc_annotation_paths)
    batchsize = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batchsize):
        end = int(np.min([start+batchsize,size]))
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        batch_new_voc_image_paths =  new_voc_image_paths[start:end]
        batch_new_voc_annotation_paths = new_voc_annotation_paths[start:end]
        pool.apply_async(batch_process_image_xml,error_callback=print_error,
                         args=(batch_voc_image_paths,batch_voc_annotation_paths,
                               batch_new_voc_image_paths,batch_new_voc_annotation_paths,class_names))
    pool.close()
    pool.join()

def is_contain_classes(xml_path,class_names):
    """
    这是判断xml文件是否包含指定目标的函数
    :param xml_path: xml文件路径
    :param class_names: 目标分类数组
    :return:
    """
    # 解析XML文件
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    # 遍历所有目标结点,判断是否存在指定目标
    flag = False
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name in class_names:
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

def process_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path,class_names):
    """
    这是完成图像复制和xml文件处理的函数
    :param voc_image_path: voc图像文件路径
    :param voc_xml_path: voc标签文件路径
    :param new_voc_image_path: voc图像文件新路径
    :param new_voc_xml_path: voc标签文件新路径
    :param class_names: 目标分类数组
    :return:
    """
    # 解析XML文件
    if os.path.exists(voc_xml_path):
        in_file = open(voc_xml_path)
        tree = ET.parse(in_file)
        root = tree.getroot()

        # 遍历所有目标结点，逐一删除不需要的目标种类的结点
        cnt = len(root.findall('object'))           # xml文件中包含的目标个数
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_names:        # 目标不在指定目标数组内则删除
                root.remove(obj)
                cnt -= 1

        if cnt > 0:         # 还存在目标则完成xml文件的复制和图像复制
            tree.write(new_voc_xml_path)
            image = cv2.imread(voc_image_path)
            cv2.imwrite(new_voc_image_path,image)

def batch_process_image_xml(batch_voc_image_paths,batch_voc_annotation_paths,
                            batch_new_voc_image_paths,batch_new_voc_annotation_paths,class_names):
    """
    這是批量处理图像和xml文件的函数
    :param batch_voc_image_paths: 批量voc图像路径数组
    :param batch_voc_annotation_paths: 批量voc标签路径数组
    :param batch_new_voc_image_paths: 批量voc图像新路径数组
    :param batch_new_voc_annotation_paths: 批量voc标签新路径数组
    :param class_names: 目标类别数组
    :return:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        voc_image_path = batch_voc_image_paths[i]
        voc_xml_path = batch_voc_annotation_paths[i]
        new_voc_image_path = batch_new_voc_image_paths[i]
        new_voc_xml_path = batch_new_voc_annotation_paths[i]
        process_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path,class_names)

def run_main():
    """
    这是主函数
    """
    '''
    # COCO2017 -> COCO2017-Person
    voc_dataset_dir = os.path.abspath("../dataset/COCO2017")
    new_voc_dataset_dir = os.path.abspath("../dataset/person/COCO2017")
    class_names = ['person']
    train_ratio = 0.8
    copy_voc_dataset(voc_dataset_dir,new_voc_dataset_dir,class_names,train_ratio)
    '''

    # VOC07+12 -> VOC07+12-Person
    voc_dataset_dir = os.path.abspath("../origin_dataset/VOC07+12")
    new_voc_dataset_dir = os.path.abspath("../dataset/person/VOC07+12")
    class_names = ['person']
    train_ratio = 0.8
    copy_voc_dataset(voc_dataset_dir, new_voc_dataset_dir, class_names, train_ratio)

if __name__ == '__main__':
    run_main()
