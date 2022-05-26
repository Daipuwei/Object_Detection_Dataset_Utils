# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 10:21
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : merge_voc_dataset.py
# @Software: PyCharm

"""
    这是将多个VOC格式的数据集合并成1个VOC格式数据集的脚本
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

def merge_voc_dataset(voc_dataset_dir_list,new_voc_dataset_dir):
    """
    这是将多个VOC格式的数据集合并成1个VOC格式数据集的函数
    :param voc_dataset_dir_list: VOC数据集目录地址列表
    :param new_voc_dataset_dir: 新VOC数据集目录地址
    :return:
    """
    # 初始化新VOC数据集相关目录地址
    new_voc_image_dir = os.path.join(new_voc_dataset_dir,'JPEGImages')
    new_voc_annotation_dir = os.path.join(new_voc_dataset_dir,'Annotations')
    new_voc_main_dir = os.path.join(new_voc_dataset_dir,"ImageSets","Main")
    new_trainval_txt_path = os.path.join(new_voc_main_dir, 'trainval.txt')
    new_train_txt_path = os.path.join(new_voc_main_dir, 'train.txt')
    new_val_txt_path = os.path.join(new_voc_main_dir, 'val.txt')
    if not os.path.exists(new_voc_image_dir):
        os.makedirs(new_voc_image_dir)
    if not os.path.exists(new_voc_annotation_dir):
        os.makedirs(new_voc_annotation_dir)
    if not os.path.exists(new_voc_main_dir):
        os.makedirs(new_voc_main_dir)

    # 遍历所有VOC数据集，获取所有图像及其XML标签文件地址
    batch_voc_image_paths = []
    batch_voc_annotation_paths = []
    batch_new_voc_image_paths = []
    batch_new_voc_annotation_paths = []
    for voc_dataset_dir in voc_dataset_dir_list:
        _,voc_dataset_name = os.path.split(voc_dataset_dir)
        voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
        voc_annotation_dir = os.path.join(voc_dataset_dir, 'Annotations')
        voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")
        trainval_txt_path = os.path.join(voc_main_dir, 'trainval.txt')
        train_txt_path = os.path.join(voc_main_dir, 'train.txt')
        val_txt_path = os.path.join(voc_main_dir, 'val.txt')
        # 遍历文件夹，生成图像和XML文件夹路径
        for image_name in os.listdir(voc_image_dir):
            fname,ext = os.path.splitext(image_name)
            batch_voc_image_paths.append(os.path.join(voc_image_dir,image_name))
            batch_voc_annotation_paths.append(os.path.join(voc_annotation_dir,fname+".xml"))
            batch_new_voc_image_paths.append(os.path.join(new_voc_image_dir,voc_dataset_name+"_"+fname+".jpg"))
            batch_new_voc_annotation_paths.append(os.path.join(new_voc_annotation_dir,voc_dataset_name+"_"+fname+".xml"))
        # 复制txt文件夹
        for new_txt_pxth,txt_path in zip([new_trainval_txt_path,new_train_txt_path,new_val_txt_path],
                                         [trainval_txt_path,train_txt_path,val_txt_path]):
            image_name_list = []
            with open(txt_path,'r') as f:
                for line in f.readlines():
                    image_name_list.append(line.strip())
            with open(new_txt_pxth,'a') as f:
                for image_name in image_name_list:
                    fname,ext = os.path.splitext(image_name)
                    f.write(voc_dataset_name+"_"+fname+"\n")

    # 多线程合并VOC数据集
    size = len(batch_voc_image_paths)
    batch_szie = size // (cpu_count() -1)
    pool = Pool(processes=cpu_count() -1)
    # batch_szie = size // 4
    # pool = Pool(processes=4)
    for start in np.arange(0,size,batch_szie):
        end = int(np.min([start+batch_szie,size]))
        pool.apply_async(batch_copy_image_xml,callback=print_error,
                         args=(batch_voc_image_paths[start:end],batch_voc_annotation_paths[start:end],
                               batch_new_voc_image_paths[start:end],batch_new_voc_annotation_paths[start:end]))
    pool.close()
    pool.join()

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def copy_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path):
    """
    这是完成图像和xml文件复制的函数
    :param voc_image_path: voc图像文件路径
    :param voc_xml_path: voc标签文件路径
    :param new_voc_image_path: voc图像文件新路径
    :param new_voc_xml_path: voc标签文件新路径
    :return:
    """
    # 解析XML文件
    in_file = open(voc_xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()

    # 复制图像和XML文件
    tree.write(new_voc_xml_path)
    in_file.close()
    image = cv2.imread(voc_image_path)
    cv2.imwrite(new_voc_image_path,image)
    del image

def batch_copy_image_xml(batch_voc_image_paths,batch_voc_annotation_paths,
                         batch_new_voc_image_paths,batch_new_voc_annotation_paths):
    """
    這是批量复制图像和xml文件的函数
    :param batch_voc_image_paths: 批量voc图像路径数组
    :param batch_voc_annotation_paths: 批量voc标签路径数组
    :param batch_new_voc_image_paths: 批量voc图像新路径数组
    :param batch_new_voc_annotation_paths: 批量voc标签新路径数
    :return:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        voc_image_path = batch_voc_image_paths[i]
        voc_xml_path = batch_voc_annotation_paths[i]
        new_voc_image_path = batch_new_voc_image_paths[i]
        new_voc_xml_path = batch_new_voc_annotation_paths[i]
        copy_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path)


def run_main():
    """
    这是主函数
    """
    voc_dataset_dir_list = [os.path.abspath("../dataset/Cityscapes"),
                            os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.01"),
                            os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.02"),
                            os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.005"),
                            os.path.abspath("../dataset/BDD100k"),
                            os.path.abspath("../dataset/KITTI")]
    new_voc_dataset_dir = os.path.abspath("../dataset/VOC_Ours")
    merge_voc_dataset(voc_dataset_dir_list,new_voc_dataset_dir)

if __name__ == '__main__':
    run_main()
