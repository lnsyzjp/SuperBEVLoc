import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tools.utils.read_bin import *
'''
imz,imi = gen_bev_img(bin_file)
'''
def scale_to_255(a, min, max, dtype=np.uint8):
    return ((a - min) / float(max - min) * 255).astype(dtype)
def gen_bev_img(bin_file):
    pointcloud = bin_file
    # 设置鸟瞰图范围
    side_range = (-40, 40)  # 左右距离
    fwd_range = (-40, 40)  # 后前距离

    x_points = pointcloud[:, 0]
    y_points = pointcloud[:, 1]
    z_points = pointcloud[:, 2]
    i_points = pointcloud[:, 3]

    # 获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    i_points = i_points[indices]

    # res = 0.1  # 分辨率0.05m
    res = 0.2  # 分辨率0.1m
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    # print(x_img.min(), x_img.max(), y_img.min(), x_img.max())

    # 填充像素值
    # height_range = (-2, 0.5)
    # pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])#clip截取部分设为指定值
    pixel_valuez = scale_to_255(z_points, np.min(z_points), np.max(z_points))
    pixel_valuei = scale_to_255(i_points, np.min(i_points), np.max(i_points))
    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)

    imz = np.zeros([y_max, x_max], dtype=np.uint8)
    imi = np.zeros([y_max, x_max], dtype=np.uint8)

    imz[y_img, x_img] = pixel_valuez
    imi[y_img, x_img] = pixel_valuei

    return np.array([imz,imi])
if __name__ == "__main__":

# 点云读取
    filepath = r"/home/zjp/dataset/nclt/2012-01-08/velodyne_sync/1326030975726043.bin"
    pointcloud = load_nclt_bin(filepath)
    im = gen_bev_img(pointcloud)
    print(np.shape(im))


# imshow （灰度）
#     im2 = Image.fromarray(im)
    # im2.save('xxx.png')
    # im2.show()

# imshow （彩色）
    plt.figure()
    plt.subplot(121)
    plt.imshow(im[1], cmap="nipy_spectral", vmin=0, vmax=255)
    plt.subplot(122)
    plt.imshow(im[0], cmap="nipy_spectral", vmin=0, vmax=255)
# plt.savefig('xxx.png',bbox_inches='tight', pad_inches = -0.1) # 注意两个参数
    plt.show()
