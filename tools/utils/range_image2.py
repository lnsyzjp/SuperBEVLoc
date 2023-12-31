import copy

import numpy as np
from read_bin import *
import matplotlib.pyplot as plt

def lidar_to_2d_front_view_depth(points,
                             v_line=65,
                             h_res=360/2048.,
                             ground_indices=None
                             ):
    """
    TODO 横向的噪点是否还是需要去除？
    是否需要保留角度信息？ 不需要，那个是雷达的自带分辨率角度
    :param points:
    :param v_line:
    :param h_res:
    :return:
    """
    x_lidar = points[:, 0]  # -71~73
    y_lidar = points[:, 1]  # -21~53
    z_lidar = points[:, 2]  # -5~2.6

    # Distance relative to origin
    d = np.sqrt(x_lidar ** 2 + y_lidar ** 2 + z_lidar ** 2)

    # set ground depth to zero
    d2 = d.copy()
    if ground_indices is not None:
        d2[ground_indices] = 0

    # Convert res to Radians
    h_res_rad = np.radians(h_res)

    #  得到水平角度
    x_radians = np.arctan2(-y_lidar, x_lidar)  # -1024~1024   -3.14~3.14  ;

    angle_diff = np.abs(np.diff(x_radians))
    threshold_angle = np.radians(250)  #
    angle_diff = np.hstack((angle_diff, 0.0001)) # 补一个元素，diff少了一个
    angle_diff_mask = angle_diff > threshold_angle

    # 把角度转换为image像素坐标
    x_img = np.floor((x_radians / h_res_rad)).astype(int)
    x_img -= np.min(x_img)  # 把坐标为负数的部分做一个转移

    y_img = np.cumsum(angle_diff_mask)

    # x_max = int(360.0 / h_res) #+ 1  # 投影后图片的宽度
    x_max = 2049
    # x_img[x_img >= x_max] = 2047

    depth_map = np.zeros((v_line, x_max, 5)) #+255
    print(np.shape(depth_map))
    print(np.shape(x_lidar))
    depth_map[y_img, x_img, 0] = x_lidar
    depth_map[y_img, x_img, 1] = y_lidar
    depth_map[y_img, x_img, 2] = z_lidar
    depth_map[y_img, x_img, 3] = d
    depth_map[y_img, x_img, 4] = d2

    return depth_map

if __name__ == "__main__":
        bin_file1 = r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\kitti\000004.bin"
        bin_file2 = r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\nclt\1339759106660157.bin"
        bin_file3 = r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\mulran\1564719512892604519.bin"
        points = np.fromfile(bin_file1, dtype=np.float32).reshape([-1, 4])
        pc1 = load_kitti_bin(bin_file1)
        pc2 = load_nclt_bin(bin_file2)
        pc3 = load_mulran_bin(bin_file3)

        # lidar_2_range(pc1, 'kitti')
        # lidar_2_range(pc2,'nclt')
        # lidar_2_range(pc3, 'mulran')

        dep = lidar_to_2d_front_view_depth(points,
                                     ground_indices=None
                                     )
        plt.figure()
        plt.imshow(dep[:,:,3])
        plt.show()