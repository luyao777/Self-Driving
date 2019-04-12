# **Finding Lane Lines on the Road** 

项目文件[来源](https://github.com/udacity/CarND-LaneLines-P1)，代码借鉴于[这里](https://github.com/yajian/self-driving-car/blob/master/p1_lane_detection/line_detection.py)。

---
## **车道检测算法**

车道检测算法主要分为以下几个步骤：

1. 灰度处理
2. 高斯模糊处理
3. Canny算法进行边缘检测，提取边缘特征
4. 设定RoI区域，进行提取
5. Hough直线检测并进行划线处理

上述对图像的处理结果可以在Jupyter Notebook中查看。

---

算法示例图像
![image1](./test_images/solidWhiteRight.jpg)

---

## **所需库的安装**

注意在安装moviepy库的时候可能会提示  `imageio` 问题

*解决方法*：

conda uninstall imageio

pip install moivepy

如果出现 `NeedDownloadError: Need ffmpeg` 问题.

*解决方法*

在Ipython中执行 `imageio.plugins.ffmpeg.download()` 进行下载

