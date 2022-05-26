# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 15:41
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : mot2voc.py
# @Software: PyCharm

"""
    这是将MOT数据集转换为VOC数据集格式的脚本，根据开发需要自行修改
    mot_dataste_dir、voc_dataset_dir和type即可。

    其中:
        - mot_dataste_dir代表原始MOT数据集目录；
        - voc_dataset_dir代表VOC数据集格式的MOT数据集目录；
        - type代表数据集类型，候选值有‘all’、‘train’和‘test’，
          ‘all’代表转化全部数据，‘train’代表转化训练数据集，‘test’代表转化测试数据集；
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from pascal_voc_writer import Writer

from multiprocessing import Pool
from multiprocessing import cpu_count

def mot2voc(mot_dataste_dir,voc_dataset_dir,type='train'):
    """
    这是利用多线程将MOT数据集转换为VOC数据集的函数
    Args:
        mot_dataste_dir: MOT数据集文件夹路径
        voc_dataset_dir: VOC数据集文件夹路径
        type: 类型，默认为'train',候选值有['train','test'，'all']
    Returns:
    """
    # 初始化VOC数据集相关路径
    voc_image_dir = os.path.join(voc_dataset_dir,'JPEGImages')
    voc_annotation_dir = os.path.join(voc_dataset_dir,"Annotations")
    voc_main_dir = os.path.join(voc_dataset_dir,'ImageSets',"Main")
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    if not os.path.exists(voc_annotation_dir):
        os.makedirs(voc_annotation_dir)
    if not os.path.exists(voc_main_dir):
        os.makedirs(voc_main_dir)

    mot_annotations = {}
    voc_image_paths = []
    voc_annotation_paths = []
    cnt = 0
    # 初始化训练集或者训练测试集的初始化MOT图像及其标签，VOC图像及其标签
    if type == 'train' or type == 'all':
        # 初始化MOT图像及其标签
        sub_dataset_dir = os.path.join(mot_dataste_dir, 'train')
        #print(sub_dataset_dir)
        for imageset_name in os.listdir(sub_dataset_dir):
            sub_imageset_dir = os.path.join(sub_dataset_dir,imageset_name,'img1')
            sub_label_txt_path = os.path.join(sub_dataset_dir,imageset_name,"gt",'gt.txt')
            #print(sub_imageset_dir,sub_label_txt_path)
            with open(sub_label_txt_path,'r') as f:
                for line in f.readlines():
                    strs = line.strip().split(",")
                    mot_image_path = os.path.join(sub_imageset_dir,"{0:06d}.jpg".format(int(strs[0])))
                    if mot_image_path not in mot_annotations.keys():
                        mot_annotations[mot_image_path] = []
                        voc_image_paths.append(os.path.join(voc_image_dir,"{0:06d}.jpg".format(cnt)))
                        voc_annotation_paths.append(os.path.join(voc_annotation_dir,"{0:06d}.xml".format(cnt)))
                        cnt += 1
                    x1 = int(round(float(strs[2])))
                    y1 = int(round(float(strs[3])))
                    w = int(round(float(strs[4])))
                    h = int(round(float(strs[3])))
                    x2 = x1+w
                    y2 = y1+h
                    mot_annotations[mot_image_path].append(['person',x1,y1,x2,y2])
    # 初始化测试集或者训练测试集的初始化MOT图像及其标签，VOC图像及其标签
    if type == 'test' or type == 'all':
        # 初始化MOT图像及其标签
        sub_dataset_dir = os.path.join(mot_dataste_dir, 'test')
        for imageset_name in os.listdir(sub_dataset_dir):
            sub_imageset_dir = os.path.join(sub_dataset_dir,imageset_name,'img1')
            sub_label_txt_path = os.path.join(sub_dataset_dir, imageset_name, "det", 'det.txt')
            with open(sub_label_txt_path,'r') as f:
                for line in f.readlines():
                    strs = line.strip().split(",")
                    mot_image_path = os.path.join(sub_imageset_dir,"{0:06d}.jpg".format(int(strs[0])))
                    if mot_image_path not in mot_annotations.keys():
                        mot_annotations[mot_image_path] = []
                        voc_image_paths.append(os.path.join(voc_image_dir,"{0:06d}.jpg".format(cnt)))
                        voc_annotation_paths.append(os.path.join(voc_annotation_dir,"{0:06d}.xml".format(cnt)))
                        cnt += 1
                    x1 = int(round(float(strs[2])))
                    y1 = int(round(float(strs[3])))
                    w = int(round(float(strs[4])))
                    h = int(round(float(strs[3])))
                    x2 = x1+w
                    y2 = y1+h
                    mot_annotations[mot_image_path].append(['person',x1,y1,x2,y2])
    mot_image_paths = np.array(list(mot_annotations.keys()))
    voc_image_paths = np.array(voc_image_paths)
    voc_annotation_paths = np.array(voc_annotation_paths)

    # 多线程生成VOC数据集
    size = len(voc_image_paths)
    batch_size = size // (cpu_count()-1)
    #print(size,cpu_count()-1,batch_size)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_voc_image_paths = voc_image_paths[start:end]
        batch_voc_annotation_paths = voc_annotation_paths[start:end]
        batch_mot_image_paths = mot_image_paths[start:end]
        pool.apply_async(batch_generate_voc_image_annotations,callback=print_error,
                         args=(batch_mot_image_paths,batch_voc_image_paths,batch_voc_annotation_paths,mot_annotations))
    pool.close()
    pool.join()

def batch_generate_voc_image_annotations(batch_mot_image_paths,batch_voc_image_paths,batch_voc_annotation_paths,mot_annotations):
    """
    这是批量生成VOC数据集的图像及其标签的函数
    Args:
        batch_mot_image_paths: 批量MOT图像路径列表
        batch_voc_image_paths: 批量VOC图像路径列表
        batch_voc_annotation_paths: 批量VOC标签路径列表
        mot_annotations: MOT图像标签字典
    Returns:
    """
    size = len(batch_voc_image_paths)
    for i in tqdm(np.arange(size)):
        generate_voc_image_annotation(batch_mot_image_paths[i],batch_voc_image_paths[i],
                                      batch_voc_annotation_paths[i],mot_annotations)

def generate_voc_image_annotation(mot_image_path,voc_image_path,voc_annotation_path,mot_annotations):
    """
    这是生成VOC数据集的图像及其标签的函数
    Args:
        mot_image_path: MOT图像路径
        voc_image_path: VOC图像路径
        voc_annotation_path: VOC标签路径
        mot_annotations: MOT图像标签字典
    Returns:
    """
    # 复制图像
    image = cv2.imread(mot_image_path)
    h,w,c = np.shape(image)
    if c == 1:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    if c == 4:
        image = cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
    cv2.imwrite(voc_image_path,image)

    # 生成XML文件
    writer = Writer(voc_annotation_path,w,h)
    for name,xmin,ymin,xmax,ymax in mot_annotations[mot_image_path]:
        writer.addObject(name,xmin,ymin,xmax,ymax)
    writer.save(voc_annotation_path)

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
    # MOT15
    mot_dataste_dir = os.path.abspath("../MOT15")
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/MOT15")
    type='train'
    print("MOT15 --> VOC Start")
    mot2voc(mot_dataste_dir, voc_dataset_dir, type)
    print("MOT15 --> VOC Finish")

    # MOT16
    mot_dataste_dir = os.path.abspath("../MOT16")
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/MOT16")
    type='train'
    print("MOT16 --> VOC Start")
    mot2voc(mot_dataste_dir, voc_dataset_dir, type)
    print("MOT16 --> VOC Finish")

    # MOT17
    mot_dataste_dir = os.path.abspath("../MOT17")
    voc_dataset_dir = os.path.abspath("../dataset/Person-NonCar/MOT17")
    type='train'
    print("MOT17 --> VOC Start")
    mot2voc(mot_dataste_dir, voc_dataset_dir, type)
    print("MOT17 --> VOC Finish")

if __name__ == '__main__':
    run_main()
