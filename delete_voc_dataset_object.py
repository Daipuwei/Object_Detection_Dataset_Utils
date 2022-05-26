# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 12:50
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : delete_voc_dataset_object.py
# @Software: PyCharm

"""
    这是利用多线程改变VOC数据集中目标分类标签的脚本
"""

import os
import numpy as np
import xml.etree.ElementTree as ET

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

def delete_voc_dataset_object(voc_dataset_dir,save_object_names):
    """
    这是改变VOC数据集中目标分类的函数
    Args:
        voc_dataset_dir: VOC数据集文件夹路径
        save_object_names: 保留目标名称列表
    Returns:
    """
    # 初始化XML标签文件数组
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    voc_annotation_paths = []
    for annotation_name in os.listdir(voc_annotation_dir):
        voc_annotation_paths.append(os.path.join(voc_annotation_dir,annotation_name))
    voc_annotation_paths = np.array(voc_annotation_paths)

    # 多线程删除目标名称
    size = len(voc_annotation_paths)
    batch_size = size // (cpu_count()-1)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        pool.apply_async(batch_delete_voc_object,callback=print_error,
                         args=(batch_voc_annotation_paths,save_object_names))
    pool.close()
    pool.join()
    #batch_delete_voc_object(voc_annotation_paths,save_object_names)

def delete_voc_object(voc_annotation_path,save_object_names):
    """
    这是删除目标标签的函数
    Args:
        voc_annotation_path: VOC数据集XML文件路径
        save_object_names: 保留目标名称列表
    Returns:
    """
    with open(voc_annotation_path, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for obj in list(root.findall('object')):
            name = obj.find('name').text
            #print(name)
            # 目标分类名称不在保留名称列表则删除节点
            if name not in save_object_names:
                root.remove(obj)
        tree.write(voc_annotation_path)

def batch_delete_voc_object(batch_voc_annotation_paths,save_object_names):
    """
    这是批量删除目标标签的函数
    Args:
        batch_voc_annotation_paths: 批量VOC数据集XML文件路径数组
        save_object_names: 原始目标名称数组
    Returns:
    """
    # 遍历所有XML标签文件，重命名目标标签
    size = len(batch_voc_annotation_paths)
    for i in tqdm(np.arange(size)):
        delete_voc_object(batch_voc_annotation_paths[i],save_object_names)

def print_error(value):
    """
    定义错误回调函数
    :param value:
    :return:
    """
    print("error: ", value)

def run_main():
    """
    这是主函数
    """
    # BDD100k
    voc_dataset_dir = os.path.abspath("../dataset/BDD100k")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Cityscapes
    voc_dataset_dir = os.path.abspath("../dataset/Cityscapes")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Foggy_Cityscapes_beta=0.005
    voc_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.01")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Foggy_Cityscapes_beta=0.01
    voc_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.02")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # Foggy_Cityscapes_beta=0.02
    voc_dataset_dir = os.path.abspath("../dataset/Foggy_Cityscapes_beta_0.005")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

    # KITTI
    voc_dataset_dir = os.path.abspath("../dataset/KITTI")
    save_object_names = ['person','rider']
    delete_voc_dataset_object(voc_dataset_dir,save_object_names)

if __name__ == '__main__':
    run_main()
