# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 11:21
# @Author  : DaiPuWei
# @Email   : 771830171@qq.com
# @File    : video2voc_dataset.py
# @Software: PyCharm

"""
    这是利用多线程对视频文件进行切割并转换为VOC格式数据集的脚本
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

def video2voc_dataset(video_paths,voc_dataset_save_dir,bin=1):
    """
    这是将视频集进行切割并转换成VOC格式数据集的函数
    Args:
        video_paths: 视频文件路径数组
        voc_dataset_save_dir: VOC数据集保存文件夹
        bin: 间隔时长，默认为1s
    Returns:
    """
    # 遍历视频，初始化VOC数据集文件夹路径
    voc_dataset_dirs = []
    for video_path in video_paths:
        _,video_name = os.path.split(video_path)
        fname,ext = os.path.splitext(video_name)
        voc_dataset_dirs.append(os.path.join(voc_dataset_save_dir,fname))

    # 多线切割视频并生成VOC数据集
    #print(video_paths)
    #print(voc_dataset_dirs)
    size = len(video_paths)
    batch_size = size // (cpu_count()-1)
    if batch_size == 0:
        batch_size = 1
    #print(size,batch_size)
    pool = Pool(processes=cpu_count()-1)
    for start in np.arange(0,size,batch_size):
        end = int(np.min([start+batch_size,size]))
        batch_video_paths = video_paths[start:end]
        batch_voc_dataset_dirs = voc_dataset_dirs[start:end]
        pool.apply_async(batch_videos2voc_datasets,callback=print_error,
                         args=(batch_video_paths,batch_voc_dataset_dirs,bin))
    pool.close()
    pool.join()

def batch_videos2voc_datasets(batch_video_paths,batch_voc_dataset_dirs,bin=1):
    """
    这是将批量视频进行切割并转换VOC格式数据集的函数
    Args:
        batch_video_paths: 批量视频文件名路径数组
        batch_voc_dataset_dirs: 批量VOC数据集文件夹路径数组
        bin: 间隔时长 ，默认为1s
    Returns:
    """
    for i in tqdm(np.arange(len(batch_video_paths))):
        single_video2voc_dataset(batch_video_paths[i],batch_voc_dataset_dirs[i],bin)

def single_video2voc_dataset(video_path,voc_dataset_dir,bin=1):
    """
    这是将单个视频进行切割并转换为VOC数据集的函数
    Args:
        video_path: 视频文件路径
        voc_dataset_dir: VOC数据集文件夹路径
        bin: 间隔时长，默认为1s
    Returns:
    """
    # 初始化视频名称
    _, video_name = os.path.split(video_path)
    fname, ext = os.path.splitext(video_name)
    # 初始化voc数据集路径
    voc_image_dir = os.path.join(voc_dataset_dir,"JPEGImages")
    voc_annotatiion_dir = os.path.join(voc_dataset_dir,"Annotations")
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    if not os.path.exists(voc_annotatiion_dir):
        os.makedirs(voc_annotatiion_dir)

    # 读取视频进行抽帧保存
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_fps = int(round(vid.get(cv2.CAP_PROP_FPS)))  # 视频的fps
    #print(video_fps)
    frame_bin = video_fps*bin               # 帧间隔
    cnt = 0
    index = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            if cnt % frame_bin == 0:
                # 图像预处理
                h = frame.shape[0]
                w = frame.shape[1]
                cv2.imwrite(os.path.join(voc_image_dir, fname + "_frame_{0:06d}.jpg".format(index)),frame)
                index += 1
            cnt += 1
        else:
            break
    vid.release()

def print_error(value):
    """
    定义错误回调函数
    Args:
        value:
    Returns:
    """
    print("error: ", value)

def run_main():
    """
    这是主函数
    """
    video_paths = [os.path.abspath("./1.mp4"),
                   os.path.abspath("./2.mp4"),
                   os.path.abspath("./3.mp4")]
    voc_dataset_save_dir = os.path.abspath("./VOCdevkit")
    bin = 1
    video2voc_dataset(video_paths,voc_dataset_save_dir,bin)


if __name__ == '__main__':
    run_main()
