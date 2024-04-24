# -*- coding: utf-8 -*-
# @Time    : 2024/4/22 下午11:37
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : labelme2voc.py
# @Software: PyCharm

"""
    这是将Labelme数据集转换为VOC数据集的脚本
"""

import os
import json
import shutil
import numpy as np
from tqdm import tqdm
from pascal_voc_writer import Writer
from multiprocessing import Pool,cpu_count

def labelme2voc(labelme_dataset_dir,voc_dataset_dir,choices=["train","val"]):
    """
    这是Labelme数据集转VOC数据集的函数
    Args:
        labelme_dataset_dir: labelme数据集地址
        voc_dataset_dir: voc数据集地址
        choices: 子集列表，默认为["train","val"]
    Returns:
    """
    # 初始化Labelme数据集相关路径
    labelme_image_dir = os.path.join(labelme_dataset_dir,'images')

    # 初始化VOC数据集相关路径
    _, voc_dataset_name = os.path.split(voc_dataset_dir)
    voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    if not os.path.exists(voc_annotation_dir):
        os.makedirs(voc_annotation_dir)
    if not os.path.exists(voc_main_dir):
        os.makedirs(voc_main_dir)

    # 初始化数据集文件路径数组
    labelme_image_names = []
    labelme_image_paths = []
    labelme_json_paths = []
    voc_image_paths = []
    voc_annotation_paths = []
    cnt = 0
    for image_name in os.listdir(labelme_image_dir):
        # json文件略过
        if image_name.endswith(".json"):
            continue
        fname,ext = os.path.splitext(image_name)
        labelme_image_names.append(fname)
        labelme_image_paths.append(os.path.join(labelme_image_dir,image_name))
        labelme_json_paths.append(os.path.join(labelme_image_dir, "{0}.json".format(fname)))
        voc_image_paths.append(os.path.join(voc_image_dir,"{0}_{1:07d}.jpg".format(voc_dataset_name, cnt)))
        voc_annotation_paths.append(os.path.join(voc_annotation_dir,"{0}_{1:07d}.xml".format(voc_dataset_name, cnt)))
        cnt += 1
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    labelme_image_paths = np.array(labelme_image_paths)
    labelme_json_paths = np.array(labelme_json_paths)

    # 生成voc子集列表
    print("开始生成VOC数据集子集列表")
    for choice in choices:
        voc_txt_path = os.path.join(voc_main_dir,"{}.txt".format(choice))
        labelme_txt_path = os.path.join(labelme_dataset_dir,"{}.txt".format(choice))
        with open(voc_txt_path,'w',encoding='utf-8') as g:
            with open(labelme_txt_path,'r',encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    labelme_fname = line.strip()
                    if labelme_fname in labelme_image_names:
                        index = labelme_image_names.index(labelme_fname)
                        _,voc_image_name = os.path.split(voc_image_paths[index])
                        voc_fname,_ = os.path.splitext(voc_image_name)
                        g.write(voc_fname+"\n")
    print("结束生成VOC数据集子集列表")

    print("开始多线程处理Labelme数据集图像及其标签并转换为VOC格式")
    size = len(labelme_image_paths)
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
    for start in np.arange(0, size, batch_size):
        end = int(np.min([start + batch_size, size]))
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        batch_labelme_image_paths = labelme_image_paths[start:end]
        batch_labelme_json_paths = labelme_json_paths[start:end]
        pool.apply_async(process_batch_images_labels, error_callback=print_error,
                         args=(batch_labelme_image_paths,batch_labelme_json_paths,
                               batch_voc_image_paths,batch_voc_annotation_paths,))
    pool.close()
    pool.join()
    print("结束多线程处理")

def process_batch_images_labels(batch_labelme_image_paths,batch_labelme_json_paths,batch_voc_image_paths,batch_voc_annotation_paths):
    """
    这是将批量Labeme数据集图像及其JSOn标签转换为VOC数据集格式的函数
    Args:
        batch_labelme_image_paths: 批量Labelme数据集图像文件路径数组
        batch_labelme_json_paths: 批量Labelme数据集Json标签文件路径数组
        batch_voc_image_paths: 批量VOC数据集图像文件路径数组
        batch_voc_annotation_paths: 批量VOC数据集XML标签文件路径数组
    Returns:
    """
    size = len(batch_labelme_image_paths)
    for i in tqdm(np.arange(size)):
        process_single_image_label(batch_labelme_image_paths[i],batch_labelme_json_paths[i],
                                   batch_voc_image_paths[i],batch_voc_annotation_paths[i])
def process_single_image_label(labelme_image_path,labelme_json_path,voc_image_path,voc_annotation_path):
    """
    这是将单张Lableme数据集图像及其Json标签转换为VOC数据集格式的函数
    Args:
        labelme_image_path: Labelme数据集图像文件路径
        labelme_json_path: Labelme数据集Json标签文件路径
                voc_image_path: VOC数据集图像文件路径
        voc_annotation_path: VOC数据集XML标签文件路径
    Returns:
    """
    # 复制图像
    shutil.copy(labelme_image_path,voc_image_path)

    # 读取labelme的json标签
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        # 读取json文件中的检测标签
        json_data = json.load(f)
        image_h = json_data["imageHeight"]
        image_w = json_data["imageWidth"]
        writer = Writer(voc_image_path,image_w,image_h)
        for shape in json_data['shapes']:
            x1,y1,x2,y2 = np.reshape(shape['points'],(-1,4))[0]
            cls_name = shape['label']
            writer.addObject(cls_name,x1,y1,x2,y2)
        writer.save(voc_annotation_path)

def print_error(value):
    """
    定义错误回调函数
    Args:
        value: 出错误值
    Returns:
    """
    print("error: ", value)



def run_main():
    """
    这是主函数
    """
    # COCO2017
    labelme_dataset_dir = "/home/dpw/deeplearning/dataset/labelme/coco2017"
    voc_dataset_dir = "/home/dpw/deeplearning/dataset/voc1/coco2017"
    choices = ["train", "val"]
    labelme2voc(labelme_dataset_dir,voc_dataset_dir,choices)

    # VOC2007
    labelme_dataset_dir = "/home/dpw/deeplearning/dataset/labelme/voc2007"
    voc_dataset_dir = "/home/dpw/deeplearning/dataset/voc1/voc2007"
    choices = ["train", "val","test"]
    labelme2voc(labelme_dataset_dir,voc_dataset_dir,choices)

    # VOC2012
    labelme_dataset_dir = "/home/dpw/deeplearning/dataset/labelme/voc2012"
    voc_dataset_dir = "/home/dpw/deeplearning/dataset/voc1/voc2012"
    choices = ["train", "val"]
    labelme2voc(labelme_dataset_dir,voc_dataset_dir,choices)


if __name__ == '__main__':
    run_main()