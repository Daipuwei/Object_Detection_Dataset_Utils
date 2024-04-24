# -*- coding: utf-8 -*-
# @Time    : 2024/4/22 下午10:27
# @Author  : DaiPuWei
# @Email   : daipuwei@qq.com
# @File    : voc2labelme.py
# @Software: PyCharm

"""
    这是将VOC数据集转换为Labelme数据集的脚本
"""

import os
import json
import shutil
import labelme
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from multiprocessing import Pool,cpu_count

def voc2labelme(voc_dataset_dir,labelme_dataset_dir,choices=["train","val"]):
    """
    这是VOC数据集转Labelme数据集的函数
    Args:
        voc_dataset_dir: voc数据集地址
        labelme_dataset_dir: labelme数据集地址
        choices: 子集列表，默认为["train","val"]
    Returns:
    """
    # 初始化VOC数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
    if not os.path.exists(voc_image_dir):
        voc_image_dir = os.path.join(voc_dataset_dir,"images")
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    voc_main_dir = os.path.join(voc_dataset_dir,"ImageSets","Main")

    # 初始化Labelme数据集相关路径
    _,labelme_dataset_name = os.path.split(labelme_dataset_dir)
    labelme_image_dir = os.path.join(labelme_dataset_dir,'images')
    if not os.path.exists(labelme_image_dir):
        os.makedirs(labelme_image_dir)

    # 初始化数据集文件路径数组
    voc_image_names = []
    voc_image_paths = []
    voc_annotation_paths = []
    labelme_image_paths = []
    labelme_json_paths = []
    cnt = 0
    for image_name in os.listdir(voc_image_dir):
        fname,ext = os.path.splitext(image_name)
        voc_image_names.append(fname)
        voc_image_paths.append(os.path.join(voc_image_dir,image_name))
        voc_annotation_paths.append(os.path.join(voc_annotation_dir,"{0}.xml".format(fname)))
        labelme_image_paths.append(os.path.join(labelme_image_dir,"{0}_{1:07d}.jpg".format(labelme_dataset_name, cnt)))
        labelme_json_paths.append(os.path.join(labelme_image_dir, "{0}_{1:07d}.json".format(labelme_dataset_name, cnt)))
        cnt += 1
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)
    labelme_image_paths = np.array(labelme_image_paths)
    labelme_json_paths = np.array(labelme_json_paths)

    # 生成labelme子集列表
    print("开始生成Labelme子集列表")
    for choice in choices:
        voc_txt_path = os.path.join(voc_main_dir,"{}.txt".format(choice))
        labelme_txt_path = os.path.join(labelme_dataset_dir,"{}.txt".format(choice))
        with open(labelme_txt_path,'w',encoding='utf-8') as g:
            with open(voc_txt_path,'r',encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    voc_fname = line.strip()
                    if voc_fname in voc_image_names:
                        index = voc_image_names.index(voc_fname)
                        _,labelme_image_name = os.path.split(labelme_image_paths[index])
                        labelme_fname,_ = os.path.splitext(labelme_image_name)
                        g.write(labelme_fname+"\n")
    print("结束生成Labelme子集列表")

    print("开始多线程处理VOC图像及其标签并转换为Labelme格式")
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
    for start in np.arange(0, size, batch_size):
        end = int(np.min([start + batch_size, size]))
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        batch_labelme_image_paths = labelme_image_paths[start:end]
        batch_labelme_json_paths = labelme_json_paths[start:end]
        pool.apply_async(process_batch_images_labels, error_callback=print_error,
                         args=(batch_voc_image_paths,batch_voc_annotation_paths,
                               batch_labelme_image_paths,batch_labelme_json_paths))
    pool.close()
    pool.join()
    print("结束多线程处理")

def process_batch_images_labels(batch_voc_image_paths,batch_voc_annotation_paths,batch_labelme_image_paths,batch_labelme_json_paths):
    """
    这是将批量VOC数据集图像及其XML标签转换为labelme数据集格式的函数
    Args:
        batch_voc_image_paths: 批量VOC数据集图像文件路径数组
        batch_voc_annotation_paths: 批量VOC数据集XML标签文件路径数组
        batch_labelme_image_paths: 批量Labelme数据集图像文件路径数组
        batch_labelme_json_paths: 批量Labelme数据集Json标签文件路径数组
    Returns:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        process_single_image_label(batch_voc_image_paths[i],batch_voc_annotation_paths[i],
                                   batch_labelme_image_paths[i],batch_labelme_json_paths[i])
def process_single_image_label(voc_image_path,voc_annotation_path,labelme_image_path,labelme_json_path):
    """
    这是将单张VOC数据集图像及其XML标签转换为labelme数据集格式的函数
    Args:
        voc_image_path: VOC数据集图像文件路径
        voc_annotation_path: VOC数据集XML标签文件路径
        labelme_image_path: Labelme数据集图像文件路径
        labelme_json_path: Labelme数据集Json标签文件路径
    Returns:
    """
    # 复制图像
    _,labelme_image_name = os.path.split(labelme_image_path)
    shutil.copy(voc_image_path,labelme_image_path)

    # 解析xml文件,写入json
    objects,(h,w) = parse_xml(voc_annotation_path)

    # 将目标框写入json文件
    json_data = {"version": labelme.__version__,
                 "flags": {},
                 "shapes": [],
                 "imagePath": labelme_image_name,
                 "imageData": None,
                 "imageHeight": h,
                 "imageWidth": w}
    for cls_name,x1,y1,x2,y2 in objects:
        json_data["shapes"].append(
            {"label": cls_name,
             "points": [[x1, y1], [x2, y2]],
             "group_id": None,
             "shape_type": "rectangle",
             "flags": {}}
        )
    with open(labelme_json_path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(json_data, indent=4,
                               separators=(',', ': '), ensure_ascii=False)
        f.write(json_data)

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
    voc_dataset_dir = "/home/dpw/deeplearning/dataset/voc/coco2017"
    labelme_dataset_dir = "/home/dpw/deeplearning/dataset/labelme/coco2017"
    choices = ["train", "val"]
    voc2labelme(voc_dataset_dir,labelme_dataset_dir,choices)

    # VOC2007
    voc_dataset_dir = "/home/dpw/deeplearning/dataset/voc/voc2007"
    labelme_dataset_dir = "/home/dpw/deeplearning/dataset/labelme/voc2007"
    choices = ["train", "val","test"]
    voc2labelme(voc_dataset_dir,labelme_dataset_dir,choices)

    # VOC2012
    voc_dataset_dir = "/home/dpw/deeplearning/dataset/voc/voc2012"
    labelme_dataset_dir = "/home/dpw/deeplearning/dataset/labelme/voc2012"
    choices = ["train", "val"]
    voc2labelme(voc_dataset_dir,labelme_dataset_dir,choices)


if __name__ == '__main__':
    run_main()
