import copy

import numpy as np
from read_bin import *
import matplotlib.pyplot as plt
'''
v_res: (float)
            vertical resolution of the lidar sensor used.
h_res: (float)
            horizontal resolution of the lidar sensor used.
v_fov: (tuple of two floats)
            (minimum_negative_angle, max_positive_angle)
'''

def lidar_2_range(points,data_name):
    '''
    velodyne HDL 32E
    视场角（垂直）：+10.67°至-30.67°；
    　　角分辨率（垂直）：1.33°；
    　　角分辨率（水平/方位）：0.1°–0.4°；
    '''
    if data_name == 'kitti':
        #velodyne HDL 64E
        v_res = 0.4  # vertical res
        h_res = 0.35  # horizontal resolution (assuming 20Hz setting)
        v_fov = (-24.9, 2.0)  # Field of view (-ve, +ve) along vertical axis
        y_fudge = 5  # y fudge factor for velodyne HDL 64E
    elif data_name == 'nclt':
        #velodyne HDL 32E
        v_res = 1.33  # vertical res
        h_res = 0.4  # horizontal resolution (assuming 20Hz setting)
        v_fov = (-30.67, 10.67)  # Field of view (-ve, +ve) along vertical axis
        y_fudge = 0  # y fudge factor for velodyne HDL 64E
    elif data_name == 'mulran':
        #+16.5° 至 -16.5°
        v_res = 0.35  # vertical res
        h_res = 0.35  # horizontal resolution (assuming 20Hz setting)
        v_fov = (-16.5, 16.5)  # Field of view (-ve, +ve) along vertical axis
        y_fudge = 5  # y fudge factor for velodyne HDL 64E
    else:
        print('There is a mistake')


    assert len(v_fov) == 2, "v_fov must be list/tuple of length 2"
    assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"

    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    r_lidar = points[:, 3]  # Reflectance
    # Distance relative to origin when looked from top
    d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
    # Absolute distance relative to origin
    # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

    v_fov_total = -v_fov[0] + v_fov[1]

    # Convert to Radians
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # PROJECT INTO IMAGE COORDINATES
    x_img = np.arctan2(-y_lidar, x_lidar) / h_res_rad
    y_img = np.arctan2(z_lidar, d_lidar) / v_res_rad

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
    x_img -= x_min  # Shift
    x_max = 360.0 / h_res  # Theoretical max x value after shifting

    y_min = v_fov[0] / v_res  # theoretical min y value based on sensor specs
    y_img -= y_min  # Shift
    y_max = v_fov_total / v_res  # Theoretical max x value after shifting

    y_max += y_fudge  # Fudge factor if the calculations based on
    # spec sheet do not match the range of
    # angles collected by in the data.
    print(x_max,y_max)
    print(len(x_img))
    # PLOT THE IMAGE
    cmap = "rainbow"  # Color map to use
    dpi = 200  # Image resolution
    fig, ax = plt.subplots(figsize=(x_max / dpi, y_max / dpi), dpi=dpi)
    ax.scatter(x_img, y_img, s=1, c=d_lidar, linewidths=0, alpha=1, cmap=cmap)

    ax.set_facecolor((0, 0, 0))  # Set regions with no points to black
    ax.axis('scaled')  # {equal, scaled}
    ax.xaxis.set_visible(True)  # Do not draw axis tick marks
    ax.yaxis.set_visible(True)  # Do not draw axis tick marks
    plt.xlim([0, x_max])  # prevent drawing empty space outside of horizontal FOV
    plt.ylim([0, y_max])  # prevent drawing empty space outside of vertical FOV
    plt.show()



if __name__ == "__main__":
    bin_file1 = r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\kitti\000000.bin"
    bin_file2 =r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\nclt\1339759075060772.bin"
    bin_file3 =r"C:\Users\Administrator\Desktop\MVSE-Net\code\data\mulran\1564719512892604519.bin"

    pc1 = load_kitti_bin(bin_file1)
    pc2 = load_nclt_bin(bin_file2)
    pc3 = load_mulran_bin(bin_file3)

    # lidar_2_range(pc1, 'kitti')
    lidar_2_range(pc2,'nclt')
    # lidar_2_range(pc3, 'mulran')


