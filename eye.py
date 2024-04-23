import cv2
import numpy as np
import glob
from math import *
import pandas as pd
import os
import json

K=np.array([[287.86974437,0,313.60807602],
            [0,287.86185364,201.84361122],
            [0,0,1]],dtype=np.float64)
chess_board_x_num=11
chess_board_y_num=8
chess_board_len=60

def myRPY2R_robot(x, y, z):
    Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    return R

def pose_robot(x, y, z, Tx, Ty, Tz):
    thetaX = x / 180 * pi
    thetaY = y / 180 * pi
    thetaZ = z / 180 * pi
    R = myRPY2R_robot(thetaX, thetaY, thetaZ)
    t = np.array([[Tx], [Ty], [Tz]])
    RT1 = np.column_stack([R, t])
    RT1 = np.row_stack((RT1, np.array([0,0,0,1])))
    return RT1

def get_RT_from_chessboard(img_path,chess_board_x_num,chess_board_y_num,K,chess_board_len):

    img=cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (chess_board_x_num, chess_board_y_num), None)

    corner_points=np.zeros((2,corners.shape[0]),dtype=np.float64)
    for i in range(corners.shape[0]):
        corner_points[:,i]=corners[i,0,:]

    object_points=np.zeros((3,chess_board_x_num*chess_board_y_num),dtype=np.float64)
    flag=0
    for i in range(chess_board_y_num):
        for j in range(chess_board_x_num):
            object_points[:2,flag]=np.array([(chess_board_x_num-j-1)*chess_board_len,(chess_board_y_num-i-1)*chess_board_len])
            flag+=1
    retval,rvec,tvec  = cv2.solvePnP(object_points.T,corner_points.T, K, distCoeffs=(-0.0124483,0.01346862,0.01416419,-0.00047718,-0.01581492))

    RT=np.column_stack(((cv2.Rodrigues(rvec))[0],tvec))
    RT = np.row_stack((RT, np.array([0, 0, 0, 1])))

    return RT,corner_points

def DLT(R, T, point):
    num_points = len(R)

    B = np.empty((0, 11))
    C = np.empty((0, 1))
    for i in range(num_points):
        X = T[i][0]
        Y = T[i][1]
        Z = T[i][2]
        # r1, r2, r3 = R[i][0], R[i][1], R[i][2]
        u= point[i][0][0]
        v=point[i][1][0]

        row1_B = np.array([X, Y, Z, 1, 0, 0, 0, 0, -X * u, -Y * u, -Z * u])
        row2_B = np.array([0, 0, 0, 0, X, Y, Z, 1, -X * v, -Y * v, -Z * v])

        B = np.vstack((B, row1_B, row2_B))

        row1_C = np.array([u])
        row2_C = np.array([v])

        C = np.vstack((C, row1_C, row2_C))


    B = np.array(B)

    C = np.vstack(C)


    B_transpose = np.transpose(B)
    BtB_inv = np.linalg.inv(B_transpose.dot(B))
    BtC = B_transpose.dot(C)
    L = BtB_inv.dot(BtC)
    return L


folder = r"D:/tamed/ThirdCalib/NO08/left_calib"

good_picture=[]
all_files = os.listdir(folder)
image_count = len(all_files)
for i in range(image_count):
    good_picture.append(i+1)
print(good_picture)

file_num=len(good_picture)

R_all_chess_to_cam_1=[]
T_all_chess_to_cam_1=[]
corner_point=[]
for i in good_picture:
    image_name = f"{i}_color.png"
    image_path = os.path.join(folder, image_name)

    if os.path.exists(image_path):

        RT,corner_points = get_RT_from_chessboard(image_path, chess_board_x_num, chess_board_y_num, K, chess_board_len)

    else:
        print(f"文件不存在：{image_path}")

    R_all_chess_to_cam_1.append(RT[:3,:3])
    T_all_chess_to_cam_1.append(RT[:3, 3].reshape((3,1)))
    corner_point.append(corner_points[:2,:1])

#end to base变换矩阵
file_address = 'D:/tamed/Projects/LoFTR-master/point_fisheye.json'  # Replace with your JSON file path

with open(file_address, 'r', encoding='utf-8') as json_file:
    poses = json.load(json_file)

R_all_end_to_base_1 = []
T_all_end_to_base_1 = []

for i in good_picture:
    pose = poses[i - 1]

    ax, ay, az = pose['coordinate'][3:]
    dx, dy, dz = pose['coordinate'][:3]

    RT = pose_robot(ax, ay, az, dx, dy, dz)

    R_all_end_to_base_1.append(RT[:3, :3])
    T_all_end_to_base_1.append(RT[:3, 3].reshape((3, 1)))


R,T=cv2.calibrateHandEye(R_all_end_to_base_1,T_all_end_to_base_1,R_all_chess_to_cam_1,T_all_chess_to_cam_1)#手眼标定
RT=np.column_stack((R,T))
RT = np.row_stack((RT, np.array([0, 0, 0, 1])))
print('相机相对于末端的变换矩阵为：')
print(RT)

R = RT[:3, :3]
T = RT[:3, 3]

rvec, _ = cv2.Rodrigues(R)
theta_x, theta_y, theta_z = rvec

print(f"Roll (theta_x): {theta_x},{theta_x*180/pi}")
print(f"Pitch (theta_y): {theta_y},{theta_y*180/pi}")
print(f"Yaw (theta_z): {theta_z},{theta_z*180/pi}")
print(f"Translation (T): {T}")

img_points=[]

#结果验证
for i in range(len(good_picture)):

    print('')
    print(i+1)
    RT_end_to_base=np.column_stack((R_all_end_to_base_1[i],T_all_end_to_base_1[i]))
    RT_end_to_base=np.row_stack((RT_end_to_base,np.array([0,0,0,1])))
    # print("EndToBase")
    # print(RT_end_to_base)

    RT_chess_to_cam=np.column_stack((R_all_chess_to_cam_1[i],T_all_chess_to_cam_1[i]))
    RT_chess_to_cam=np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print("ChessToCam")
    # print(RT_chess_to_cam)

    RT_cam_to_end=np.column_stack((R,T))
    RT_cam_to_end=np.row_stack((RT_cam_to_end,np.array([0,0,0,1])))
    # print(RT_cam_to_end)

    RT_chess_to_base=RT_end_to_base@RT_cam_to_end@RT_chess_to_cam#即为固定的棋盘格相对于机器人基坐标系位姿
    # RT_chess_to_base=np.linalg.inv(RT_chess_to_base)
    print("ChessToBase")
    print(RT_chess_to_base[:3,:])