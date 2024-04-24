# -*- coding: utf-8 -*-
# @Time    : 2024/4/21 下午9:22
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : append_voc_dataset_prefix.py
# @Software: PyCharm

"""
    这是给VOC数据集图像及其标签文件加上前缀的脚本
"""

import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pascal_voc_writer import Writer
from multiprocessing import Pool,cpu_count

def append_voc_dataset_prefix(voc_dataset_dir, voc_dataset_prefix):
    """
    这是给VOC数据集图像及其标签文件加上前缀的函数
    Args:
        voc_dataset_dir: VOC数据集地址
        voc_dataset_prefix: VOC数据集前缀
    Returns:
    """
    # 初始化voc数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir, 'JPEGImages')
    if not os.path.exists(voc_image_dir):
        voc_image_dir = os.path.join(voc_dataset_dir, 'images')
    voc_annotation_dir = os.path.join(voc_dataset_dir, "Annotations")
    voc_main_dir = os.path.join(voc_dataset_dir,'ImageSets',"Main")

    # 更新子集txt文件
    for choice in ["train","val","trainval","test"]:
        choice_txt_path = os.path.join(voc_main_dir, choice + '.txt')
        if not os.path.exists(choice_txt_path):
            continue
        lines = []
        with open(choice_txt_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                lines.append(line.strip())
        with open(choice_txt_path,'w',encoding='utf-8') as f:
            for line in lines:
                f.write("{0}_{1}\n".format(voc_dataset_prefix,line))

    # 初始化相关路径
    voc_image_paths = []
    voc_annotation_paths = []
    new_voc_image_paths = []
    new_voc_annotation_paths = []
    for image_name in os.listdir(voc_image_dir):
        fname,ext = os.path.splitext(image_name)
        voc_image_paths.append(os.path.join(voc_image_dir,image_name))
        voc_annotation_paths.append(os.path.join(voc_annotation_dir,fname+".xml"))
        new_voc_image_paths.append(os.path.join(voc_image_dir,"{0}_{1}".format(voc_dataset_prefix,image_name)))
        new_voc_annotation_paths.append(os.path.join(voc_annotation_dir,"{0}_{1}.xml".format(voc_dataset_prefix,fname)))
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    new_voc_image_paths = np.array(new_voc_image_paths)
    new_voc_annotation_paths = np.array(new_voc_annotation_paths)

    # 多线程处理复制图像和xml文件
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
        pool.apply_async(batch_process_image_xml,error_callback=print_error,
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

def process_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path):
    """
    这是复制VOC图像及其xml文件的函数
    Args:
        voc_image_path: voc图像文件路径
        voc_xml_path: voc标签文件路径
        new_voc_image_path: voc图像文件新路径
        new_voc_xml_path: voc标签文件新路径
    Returns:
    """
    # 复制图像
    shutil.copy(voc_image_path,new_voc_image_path)

    # xml解析复制制定类别目标
    objects,(h,w) = parse_xml(voc_xml_path)
    writer = Writer(new_voc_image_path,w,h)
    for cls_name,x1,y1,x2,y2 in objects:
        writer.addObject(cls_name,x1,y1,x2,y2)
    writer.save(new_voc_xml_path)

def batch_process_image_xml(batch_voc_image_paths,batch_voc_annotation_paths,
                            batch_new_voc_image_paths,batch_new_voc_annotation_paths):
    """
    这是批量处理VOC图像及其标签函数
    Args:
        batch_voc_image_paths: 批量voc图像路径数组
        batch_voc_annotation_paths: 批量voc标签路径数组
        batch_new_voc_image_paths: 批量voc图像新路径数组
        batch_new_voc_annotation_paths: 批量voc标签新路径数组
    Returns:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        voc_image_path = batch_voc_image_paths[i]
        voc_xml_path = batch_voc_annotation_paths[i]
        new_voc_image_path = batch_new_voc_image_paths[i]
        new_voc_xml_path = batch_new_voc_annotation_paths[i]
        process_image_xml(voc_image_path,voc_xml_path,new_voc_image_path,new_voc_xml_path)


def run_main():
    """
    这是主函数
    """


if __name__ == '__main__':
    run_main()
