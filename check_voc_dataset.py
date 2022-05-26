# -*- coding: utf-8 -*-
# @Time    : 2022/5/10 12:56
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : check_voc_dataset.py
# @Software: PyCharm

"""
    这是检查VOC数据集是否存在异常的脚本
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

def check_voc_dataset(voc_dataset_dir):
    """
    这是检查VOC数据集是否存在异常的函数
    Args:
        voc_dataset_dir: VOC数据集文件夹地址
    Returns:
    """
    # 初始化相关路径
    voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
    voc_annotatiion_dir = os.path.join(voc_dataset_dir,"Annotations")

    # 初始化xml和图像路径
    voc_image_paths = []
    voc_annotatiion_paths = []
    for image_name in os.listdir(voc_image_dir):
        fname,ext = os.path.splitext(image_name)
        voc_image_paths.append(os.path.join(voc_image_dir,image_name))
        voc_annotatiion_paths.append(os.path.join(voc_annotatiion_dir,fname+".xml"))
    voc_image_paths = np.array(voc_image_paths)
    voc_annotatiion_paths = np.array(voc_annotatiion_paths)

    # 多线程检查VOC数据集
    size = len(voc_image_paths)
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotatiion_paths[start:end]
        pool.apply_async(batch_check_images_annotations,callback=print_error,
                         args=(batch_voc_image_paths,batch_voc_annotation_paths))
    pool.close()
    pool.join()

def batch_check_images_annotations(batch_voc_image_paths,batch_voc_annotation_paths):
    """
    这是批量检查图像文件和标签文件是否一一对应的函数
    :param batch_voc_image_paths: 批量VOC图像路径数组
    :param batch_voc_annotation_paths: 批量VOC标签文件路径数组
    :return:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        check_image_annotation(batch_voc_image_paths[i],batch_voc_annotation_paths[i])

def check_image_annotation(voc_image_path,voc_annotation_path):
    """
    这是图像文件和标签文件是否对应的函数
    :param voc_image_path: VOC图像路径
    :param voc_annotation_path: VOC标签文件路径
    :return:
    """
    if not os.path.exists(voc_annotation_path):             # XML文件不存在，则删除图片
        os.remove(voc_image_path)
    else:
        if get_object_num(voc_annotation_path) == 0:        # XML文件中目标数为0，则将图片和标签全部删除
            os.remove(voc_image_path)
            os.remove(voc_annotation_path)
        else:
            image = cv2.imread(voc_image_path)
            if image is None:
                os.remove(voc_image_path)
                os.remove(voc_annotation_path)

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def get_object_num(xml_path):
    """
    这是解析XML文件目标个数的函数
    :param xml_path: XML文件路径
    :return:
    """
    tree = ET.parse(xml_path)
    return len(tree.findall('object'))


def run_main():
    """
    这是主函数
    """
    voc_dataset_dirs = [os.path.abspath("../dataset/BDD100k"),
                        os.path.abspath("../dataset/Cityscapes"),
                        os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.01"),
                        os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.02"),
                        os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.005"),
                        os.path.abspath("../dataset/KITTI")]
    for voc_dataset_dir in voc_dataset_dirs:
        check_voc_dataset(voc_dataset_dir)


if __name__ == '__main__':
    run_main()
