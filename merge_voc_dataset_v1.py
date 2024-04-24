# -*- coding: utf-8 -*-
# @Time    : 2022/4/8 10:21
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : merge_voc_dataset_v1.py
# @Software: PyCharm

"""
    这是将多个VOC格式的数据集合并成1个VOC格式数据集的脚本,根据子集对应关系进行合并
"""

import os
import cv2
import shutil
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count
from pascal_voc_writer import Writer

def merge_voc_dataset(voc_dataset_dir_list,new_voc_dataset_dir):
    """
    这是合并VOC数据集的函数
    Args:
        voc_dataset_dir_list: VOC数据集地址列表
        new_voc_dataset_dir: 合并后的VOC数据集地址
    Returns:
    """
    # 初始化新VOC数据集相关目录地址
    new_voc_image_dir = os.path.join(new_voc_dataset_dir,'JPEGImages')
    new_voc_annotation_dir = os.path.join(new_voc_dataset_dir,'Annotations')
    new_voc_main_dir = os.path.join(new_voc_dataset_dir,"ImageSets","Main")
    if not os.path.exists(new_voc_image_dir):
        os.makedirs(new_voc_image_dir)
    if not os.path.exists(new_voc_annotation_dir):
        os.makedirs(new_voc_annotation_dir)
    if not os.path.exists(new_voc_main_dir):
        os.makedirs(new_voc_main_dir)

    # 合并子集txt文件
    for choice in ['train','val','test','trainval']:
        new_choice_txt_path = os.path.join(new_voc_main_dir, '{}.txt'.format(choice))
        with open(new_choice_txt_path,'w',encoding='utf-8') as f:
            for voc_dataset_dir in voc_dataset_dir_list:
                voc_main_dir = os.path.join(voc_dataset_dir, "ImageSets", "Main")
                choice_txt_path = os.path.join(voc_main_dir, '{}.txt'.format(choice))
                if os.path.exists(choice_txt_path):
                    with open(choice_txt_path,'r',encoding='utf-8') as g:
                        for line in g.readlines():
                            f.write(line)

    # 遍历所有VOC数据集，获取所有图像及其XML标签文件地址
    voc_image_paths = []
    voc_annotation_paths = []
    new_voc_image_paths = []
    new_voc_annotation_paths = []
    for voc_dataset_dir in voc_dataset_dir_list:
        voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
        if not os.path.exists(voc_image_dir):
            voc_image_dir = os.path.join(voc_dataset_dir,"images")
        voc_annotation_dir = os.path.join(voc_dataset_dir, 'Annotations')
        # 遍历文件夹，生成图像和XML文件夹路径
        for image_name in os.listdir(voc_image_dir):
            fname,ext = os.path.splitext(image_name)
            voc_image_paths.append(os.path.join(voc_image_dir,image_name))
            voc_annotation_paths.append(os.path.join(voc_annotation_dir,fname+".xml"))
            new_voc_image_paths.append(os.path.join(new_voc_image_dir,image_name))
            new_voc_annotation_paths.append(os.path.join(new_voc_annotation_dir,fname+".xml"))
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    new_voc_image_paths = np.array(new_voc_image_paths)
    new_voc_annotation_paths = np.array(new_voc_annotation_paths)

    # 多线程合并VOC数据集
    size = len(voc_image_paths)
    if size // cpu_count() != 0:
        num_threads = cpu_count()
    elif size // (cpu_count() // 2) != 0:
        num_threads = cpu_count() // 2
    elif size // (cpu_count() // 4) != 0:
        num_threads = cpu_count() // 4
    else:
        num_threads = 1
    batch_size = size // num_threads
    pool = Pool(processes=num_threads)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        batch_new_voc_image_paths = new_voc_image_paths[start:end]
        batch_new_voc_annotation_paths = new_voc_annotation_paths[start:end]
        pool.apply_async(copy_batch_images_xmls,callback=print_error,
                         args=(batch_voc_image_paths,batch_voc_annotation_paths,
                               batch_new_voc_image_paths,batch_new_voc_annotation_paths))
    pool.close()
    pool.join()

def print_error(value):
    """
    定义错误回调函数
    Args:
        value: 出错误值
    Returns:
    """
    print("error: ", value)

def parse_xml(xml_path,class_names=None):
    """
     这是解析VOC数据集XML标签文件，获取每个目标分类与定位的函数
    Args:
        xml_path: XML标签文件路径
        class_names: 目标名称数组，默认为None
    Returns:
    """
    # 获取XML文件的根结点
    root = ET.parse(xml_path).getroot()
    h = int(root.find("size").find("height").text)
    w = int(root.find("size").find("width").text)
    # 遍历所有目标
    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        if class_names is None:
            objects.append([obj_name, int(xmin), int(ymin), int(xmax), int(ymax)])
        else:
            if obj_name in class_names:
                objects.append([obj_name, int(xmin), int(ymin), int(xmax), int(ymax)])
    return objects,(h,w)


def copy_single_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path):
    """
    这是复制单张VOC图像及其对应xml标签文件的函数
    Args:
        voc_image_path: voc图像文件路径
        voc_xml_path: voc标签文件路径
        new_voc_image_path: 新voc图像文件路径
        new_voc_xml_path: 新voc标签文件路径
    Returns:
    """
    # 复制图像
    shutil.copy(voc_image_path,new_voc_image_path)

    # 解析XML文件
    objects,(h,w) = parse_xml(voc_xml_path)

    # xml文件复制
    writer = Writer(new_voc_image_path,w,h)
    for cls_name,x1,y1,x2,y2 in objects:
        writer.addObject(cls_name,x1,y1,x2,y2)
    writer.save(new_voc_xml_path)

def copy_batch_images_xmls(batch_voc_image_paths,batch_voc_annotation_paths,
                           batch_new_voc_image_paths,batch_new_voc_annotation_paths):
    """
    这是复制批量VOC图像及其对应xml标签文件的函数
    Args:
        batch_voc_image_paths: 批量voc图像文件路径数组
        batch_voc_annotation_paths: 批量voc标签文件路径数组
        batch_new_voc_image_paths: 批量新voc图像文件路径数组
        batch_new_voc_annotation_paths: 批量新voc标签文件路径数组
    Returns:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        voc_image_path = batch_voc_image_paths[i]
        voc_xml_path = batch_voc_annotation_paths[i]
        new_voc_image_path = batch_new_voc_image_paths[i]
        new_voc_xml_path = batch_new_voc_annotation_paths[i]
        copy_single_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path)

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
