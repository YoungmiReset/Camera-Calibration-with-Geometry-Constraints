import glob
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_unit
import math
import os
import matplotlib.pyplot as plt

ratio=1.0

def get_rotation(x_,y_,z_):

    # print(math.cos(math.pi/2))
    x=float(x_/180)*math.pi
    y=float(y_/180)*math.pi
    z=float(z_/180)*math.pi
    R_x=np.array([[1, 0, 0 ],
                 [0, math.cos(x), -math.sin(x)],
                 [0, math.sin(x), math.cos(x)]])

    R_y=np.array([[math.cos(y), 0, math.sin(y)],
                 [0, 1, 0],
                 [-math.sin(y), 0, math.cos(y)]])

    R_z=np.array([[math.cos(z), -math.sin(z), 0 ],
                 [math.sin(z), math.cos(z), 0],
                 [0, 0, 1]])
    return np.dot(R_z,np.dot(R_y,R_x))
# def getFiles(file_dir,suf):
#     L=[]
#     for root, dirs, files in os.walk(file_dir):
#         for file in files:
#             if os.path.splitext(file)[1] == suf:
#                 L.append(os.path.join(root, file))
#         L.sort(key = lambda  x: int(x.split('/')[-1].split('_')[0]))
#     return L
def getFiles(file_dir, suf):
    L = []

    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == suf:
                L.append(os.path.join(root, file))

    L.sort(key=lambda x: int(os.path.split(x)[-1].split('_')[0]))

    return L

#得到相机到base坐标系的位姿转换，返回姿态和位置
def read_3d_pos(path_real):
    

    f = open(path_real, 'r', encoding='utf-8')
    ms = json.load(f)

    #相机相对于末端的位姿变换
 #    RT = np.array([[ 6.94946376e-01,7.18971222e-01 , -1.13981259e-02, 2.01649372e+01],
 # [-7.19030937e-01 ,6.94975769e-01  ,-1.78674778e-03 , 6.07239399e+01],
 # [6.63680111e-03  ,9.43729907e-03  ,9.99933443e-01  ,1.64390241e+02 ],
 # [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
 #
 #    x = 0.3588374
 #    y = -0.57658404
 #    z = -45.9735206
    RT = np.array([[7.54976899e-01,  6.55745548e-01,  2.76744346e-03,  2.00558417e+01],
 [-6.55336735e-01,  7.54642841e-01, -3.23719822e-02,  6.37783617e+00],
 [-2.33162146e-02,  2.26264914e-02,  9.99472059e-01,  3.72380160e+02],
 [ 0.00000000e+00,  0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    x = 1.71865361
    y =0.81509122
    z = -40.97016066


    camRs = []
    cam3Ds = []

    arm3ds = []
    R2 = get_rotation(x,y,z)
    t2 = RT[0:3,3].reshape(3,1)
    for m in ms:
        ti = np.array(m["coordinate"])[0:3]
        Ri = get_rotation(np.array(m["coordinate"])[3:][0],np.array(m["coordinate"])[3:][1],np.array(m["coordinate"])[3:][2])

        camR = np.dot(Ri, R2)

        cam3d = np.dot(Ri, t2)+ti.reshape(3,1)

        arm3ds.append(ti.reshape(1,3))

        camRs.append(camR)
        cam3Ds.append(cam3d.reshape(1,3))

    return np.array(camRs), np.array(cam3Ds)
#旋转角度差值
def calcAngularDistance(Rt, R):

    rotDiff = np.dot(Rt.T, R)
    if len(rotDiff.shape)<2:
        print(rotDiff)
    elif rotDiff.shape[0]!=3 or rotDiff.shape[1]!=3:
        print(rotDiff)
    trace = np.trace(rotDiff)
    cos =(float(trace) - 1.0) / 2.0
    cos = min(cos, 1)
    cos = max(cos, -1)
    theta = np.arccos(cos)

    return float(180 * theta / math.pi)

def test_K_dist(img, mtx, dist, objp, R,T):


    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    imgpoints2, _ = cv2.projectPoints(objp, R, T, mtx, dist)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(img, (8, 11))
    cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
    error2 = cv2.norm(corners, imgpoints2, cv2.NORM_L2) / len(imgpoints2)

    return error2

# 使用箱线图方法识别离群值
def remove_outliers(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return data[(data >= lower_bound) & (data <= upper_bound)]

def get_K_and_D(checkerboard, imgsPath, camrs, camts):
    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    # calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*60
    _img_shape = None
    objpoints = []
    imgpoints = []

    iss = getFiles(imgsPath, ".png")

    x = int(len(iss)*ratio)
    images = getFiles(imgsPath, ".png")[0:x]
    images_all = getFiles(imgsPath, ".png")
    gray = None


    #传统方法求所有图片的标定结果
    for ij in range(len(images_all)):
        img = cv2.imread(images_all[ij])
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCornersSB(img, (8, 11))
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    num = N_OK
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num)]
    rms0, mtx0, dist0, Rs0, Ts0 = cv2.calibrateCamera(
        objpoints[0:num],
        imgpoints[0:num],
        gray.shape[::-1],
        K.copy(),
        D,
        rvecs,
        tvecs,criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print("K",mtx0)
    print("D",dist0)
    print("rms", rms0)



    Tglobals = []
    xyzs = []
    objpoints_choice = []
    imgpoints_choice = []
    img_choice = []
    imgids_choice = []

    for ii in range(len(images_all)):
        # imgid = int(images_all [ii].split('/')[-1].split('_')[0])-1
        imgid = int(images_all[ii].split(os.path.sep)[-1].split('_')[0]) - 1

        Rboard = cv2.Rodrigues(Rs0[imgid])[0]
        Tboard = Ts0[imgid]
        Rboard_global = np.dot(camrs[imgid],Rboard)
        Tboard_global = np.dot(camrs[imgid],Tboard) + camts[imgid].reshape(3,1)
        Rg = R_unit.from_matrix(Rboard_global)
        xyz = Rg.as_euler("xyz",True)
        xyzs.append(xyz)
        Tglobals.append(Tboard_global.reshape((1,3)))

    RRR = get_rotation(np.array(xyzs).mean(axis=0)[0], np.array(xyzs).mean(axis=0)[1], np.array(xyzs).mean(axis=0)[2])
    TTT=np.array(Tglobals).mean(axis=0)


    deltaRs = []
    deltas = []
    T3_jcs=[]
    R3_jcs=[]


    for ii in range(len(images_all)):
        # imgid = int(images_all [ii].split('/')[-1].split('_')[0])-1
        imgid = int(images_all[ii].split(os.path.sep)[-1].split('_')[0]) - 1
        Rboard = cv2.Rodrigues(Rs0[imgid])[0]
        Tboard = Ts0[imgid]
        Rboard_global = np.dot(camrs[imgid],Rboard)
        Tboard_global = np.dot(camrs[imgid],Tboard) + camts[imgid].reshape(3,1)

        deltaR = calcAngularDistance(RRR, Rboard_global)

        delta = np.linalg.norm(TTT.reshape(3, 1) - Tboard_global.reshape(3, 1), axis=0)

        deltaRs.append(deltaR)
        deltas.append(delta)

        T3_c = np.dot(camrs[imgid].T, (TTT.reshape(3, 1) - camts[imgid].reshape(3, 1)))
        R3_c = np.dot(camrs[imgid].T, RRR)

        T3_jcs.append(T3_c)
        R3_jcs.append(R3_c)

    choice_num = 0

    objpoints = []
    imgpoints = []

    # print("np.array(deltas).mean(): ",np.array(deltas).mean())
    # print("np.array(deltaRs).mean(): ", np.array(deltaRs).mean())
    # np.array(deltas).mean()
    A=[]
    R=[]
    C=[]
    D=[]
    for ij in range(len(images)):


        img = cv2.imread(images[ij])
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(img, (8, 11))
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners)

        # imgid = int(images[ij].split('/')[-1].split('_')[0]) - 1
        imgid = int(images[ij].split(os.path.sep)[-1].split('_')[0]) - 1
        Rboard = cv2.Rodrigues(Rs0[imgid])[0]
        Tboard = Ts0[imgid]

        Rboard_global = np.dot(camrs[imgid], Rboard)
        Tboard_global = np.dot(camrs[imgid], Tboard) + camts[imgid].reshape(3, 1)
        deltaR = calcAngularDistance(RRR, Rboard_global)

        delta = np.linalg.norm(TTT.reshape(3, 1) - Tboard_global.reshape(3, 1), axis=0)
        # 剔除离群值
        A_filtered = remove_outliers(np.array(deltas))
        R_filtered = remove_outliers(np.array(deltaRs))

        # print("Filtered A:", A_filtered)
        # print("Filtered R:", R_filtered)

        # # 绘制箱线图
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.boxplot(A)
        # plt.title('Boxplot of A')
        # plt.subplot(1, 2, 2)
        # plt.boxplot(R)
        # plt.title('Boxplot of R')
        # plt.show()
        scale = 1.5
        if delta < np.array(deltas).mean() * scale and deltaR < np.array(deltaRs).mean() * scale:
            objpoints_choice.append(objpoints[imgid])
            imgpoints_choice.append(imgpoints[imgid])
            img_choice.append(img)
            imgids_choice.append(imgid)
            choice_num += 1
            C.append(delta[0])
            D.append(deltaR)

    print(np.array(deltas).mean())
    print(np.array(deltaRs).mean())
    print(choice_num)

    print(imgids_choice)
    N_OK = len(objpoints)
    K = np.zeros((3, 3))

    D = np.zeros((4, 1))
    num = N_OK
    # print(num)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num)]
    rms, mtx, dist, Rs, Ts = cv2.calibrateCamera(
        objpoints[0:num],
        imgpoints[0:num],
        gray.shape[::-1],
        K.copy(),
        D,
        rvecs,
        tvecs,criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    rmsc, mtxc, distc, Rsc, Tsc = cv2.calibrateCamera(
        objpoints_choice,
        imgpoints_choice,
        gray.shape[::-1],
        K.copy(),
        D,
        rvecs,
        tvecs, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    print("______________________")
    print("optimized_K",mtxc)
    print("optimized_D", distc)
    print("optimized_rms",rmsc)
    print("=======================")




    imagestest1 = getFiles(imgsPath, ".png")[x:]
    imagestest2 = getFiles(imgsPath, ".png")[0:]
    total_error_jc=0
    total_error_board = 0
    for ij in range(len(imagestest1)):
        img = cv2.imread(imagestest1[ij])

        # imgid = int(imagestest[ij].split('/')[-1].split('_')[0]) - 1
        imgid = int(imagestest1[ij].split(os.path.sep)[-1].split('_')[0]) - 1

        error_jc = test_K_dist(img, mtxc,distc, objp, R3_jcs[imgid],T3_jcs[imgid])
        total_error_jc += error_jc

        error_board = test_K_dist(img, mtx, dist, objp, R3_jcs[imgid], T3_jcs[imgid])
        total_error_board += error_board

    total_error_jc_all = 0
    total_error_board_all = 0
    for ij in range(len(imagestest2)):
        img = cv2.imread(imagestest2[ij])

        # imgid = int(imagestest[ij].split('/')[-1].split('_')[0]) - 1
        imgid = int(imagestest2[ij].split(os.path.sep)[-1].split('_')[0]) - 1

        error_jc = test_K_dist(img, mtxc, distc, objp, R3_jcs[imgid], T3_jcs[imgid])
        total_error_jc_all += error_jc

        error_board = test_K_dist(img, mtx, dist, objp, R3_jcs[imgid], T3_jcs[imgid])
        total_error_board_all += error_board

    image_folder="D:/tamed/ThirdCalib/NO23/left_calib"
    image_paths = glob.glob(image_folder + "/*.png")
    # 初始化重投影误差
    total_reprojection_error = 0.0
    num_images = 0

    # 遍历所有图片
    for image_path in image_paths:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

        if ret:
            num_images += 1
            # 定义3D点坐标(棋盘格角点)
            # objp = np.zeros((11 * 8, 3), np.float32)
            # objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)*0.06  # 乘以0.06将尺寸转换为米

            # 精细化角点位置
            corners2 = cv2.cornerSubPix(gray, corners, (11, 8), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # 计算重投影误差
            image_points, _ = cv2.projectPoints(objp, np.zeros(3), np.zeros(3), mtxc, distc)
            reprojection_error = cv2.norm(corners2, image_points, cv2.NORM_L2) / len(objp)
            total_reprojection_error += reprojection_error

    # 计算平均重投影误差
    if num_images > 0:
        avg_reprojection_error = total_reprojection_error / num_images
        print(f"Average reprojection error: {avg_reprojection_error}")
    else:
        print("No valid images found.")

    return  choice_num,mtx0,dist0,mtxc, distc,total_error_board/len(imagestest1),total_error_jc / len(imagestest1),total_error_board_all/len(imagestest2),total_error_jc_all / len(imagestest2),np.array(deltas).mean(),np.array(deltaRs).mean(),RRR,TTT


if __name__ == '__main__':

    p1_json = 'D:/tamed/Projects/LoFTR-master/point_fisheye.json'

    p1 = 'D:/tamed/ThirdCalib/realsense/RealSense_320_180/imgs_light/imgs'
    p11 = 'D:/tamed/ThirdCalib/realsense/RealSense_320_180/imgs_lightt/imgs'
    p111= 'D:/tamed/ThirdCalib/realsense/RealSense_320_180/imgs_lighttt/imgs'
    p2 = 'D:/tamed/ThirdCalib/realsense/RealSense_640_360/imgs_light/imgs'
    p22 = 'D:/tamed/ThirdCalib/realsense/RealSense_640_360/imgs_lightt/imgs'
    p222 = 'D:/tamed/ThirdCalib/realsense/RealSense_640_360/imgs_lighttt/imgs'
    p3 = 'D:/tamed/ThirdCalib/realsense/RealSense_848_480/imgs_light/imgs'
    p33 = 'D:/tamed/ThirdCalib/realsense/RealSense_848_480/imgs_lightt/imgs'
    p333 = 'D:/tamed/ThirdCalib/realsense/RealSense_848_480/imgs_lighttt/imgs'
    p4 = 'D:/tamed/ThirdCalib/realsense/RealSense_960_540/imgs_light/imgs'
    p44 = 'D:/tamed/ThirdCalib/realsense/RealSense_960_540/imgs_lightt/imgs'
    p444 = 'D:/tamed/ThirdCalib/realsense/RealSense_960_540/imgs_lighttt/imgs'
    p5 = 'D:/tamed/ThirdCalib/realsense/RealSense_1280_720/imgs_light/imgs'
    p55 = 'D:/tamed/ThirdCalib/realsense/RealSense_1280_720/imgs_lightt/imgs'
    p555 = 'D:/tamed/ThirdCalib/realsense/RealSense_1280_720/imgs_lighttt/imgs'
    p6 = 'D:/tamed/ThirdCalib/realsense/RealSense_1920_1080/imgs_light/imgs'
    p66 = 'D:/tamed/ThirdCalib/realsense/RealSense_1920_1080/imgs_lightt/imgs'
    p666 = 'D:/tamed/ThirdCalib/realsense/RealSense_1920_1080/imgs_lighttt/imgs'

    p88='D:/tamed/ThirdCalib/NO10/left_calib3'
    # p=(p1,p11,p111,p2,p22,p222,p3,p33,p333,p4,p44,p444,p5,p55,p555,p6,p66,p666)
    # p = (p33, p333, p4, p444, p5,  p55,p6,  p666)
    # p = (p3,p33,p333,p4,p44,p444,p5,p55,p555,p6,p66,p666)
    # p = ( p33, p333, p4, p44,p55, p555, p6, p66)

    KK_ori=[]
    DD_ori=[]
    KK=[]
    DD=[]
    test_error_orii=[]
    test_errorr=[]
    test_error_orii1 = []
    test_errorr1 = []
    delaa=[]
    delRR=[]
    delerrorr=[]
    delerrorr1 = []
    RR=[]
    TT=[]
    choice=[]

    camrs, camts  = read_3d_pos(p1_json)


    choice_num, K_ori, D_ori, K, D, test_error_ori, test_error, test_error_all_ori, test_error_all, dela, delR, RRR, TTT = get_K_and_D(
        (8, 11), p88, camrs, camts)
    KK_ori.append(K_ori)
    DD_ori.append(D_ori)
    KK.append(K)
    DD.append(D)
    test_error_orii.append(test_error_ori)
    test_errorr.append(test_error)
    test_error_orii1.append(test_error_all_ori)
    test_errorr1.append(test_error_all)
    delaa.append(dela)
    delRR.append(delR)
    delerrorr.append(test_error - test_error_ori)
    delerrorr1.append(test_error_all - test_error_all_ori)
    RR.append(RRR)
    TT.append(TTT)
    choice.append(choice_num)
    #
    # for p_name in p:
    #     choice_num,K_ori, D_ori, K, D ,test_error_ori, test_error,test_error_all_ori, test_error_all,dela,delR,RRR,TTT= get_K_and_D((8, 11), p_name, camrs, camts)
    #     KK_ori.append(K_ori)
    #     DD_ori.append(D_ori)
    #     KK.append(K)
    #     DD.append(D)
    #     test_error_orii.append(test_error_ori)
    #     test_errorr.append(test_error)
    #     test_error_orii1.append(test_error_all_ori)
    #     test_errorr1.append(test_error_all)
    #     delaa.append(dela)
    #     delRR.append(delR)
    #     delerrorr.append(test_error-test_error_ori)
    #     delerrorr1.append(test_error_all - test_error_all_ori)
    #     RR.append(RRR)
    #     TT.append(TTT)
    #     choice.append(choice_num)
    # print("KKKKKKKK_ori",KK_ori)
    # print("DDDDDDDD_ori", DD_ori)
    # print("KKKKKKKK",KK)
    # print("DDDDDDDD", DD)
    print("过滤后的标定图像数量",choice)
    print("delaa",delaa)
    print("delRR",delRR)
    print("RRR", RR)
    print("TTT", TT)
    print("test_error_orii",test_error_orii)
    print("test_errorr",test_errorr)
    print("delerror",delerrorr)
    print("***************所有图像作为测试集************")
    print("test_error_orii_all", test_error_orii1)
    print("test_errorr_all", test_errorr1)
    print("delerror_all", delerrorr1)
    # # plt.plot(test_error_orii, color='blue', label='test_error_orii')
    # plt.plot(delerrorr, color='green', label='test_error_improve')
    # plt.axhline(y=0, color='red', linestyle='--')  # 添加纵坐标为0处的红色虚线水平线
    # plt.xlabel('Index')
    # plt.ylabel('Error')
    # plt.title('Test Error')
    # plt.legend()
    # plt.show()

    # 从KK_ori和DD_ori中取出对应位置的数组
    FX_ori=[]
    FY_ori=[]
    CX_ori=[]
    CY_ori=[]
    K1_ori=[]
    K2_ori = []
    K5_ori = []
    for arr in KK_ori:
        fx_ori = arr[0][0]
        fy_ori = arr[1][1]
        cx_ori = arr[0][2]
        cy_ori = arr[1][2]
        FX_ori.append(fx_ori)
        FY_ori.append(fy_ori)
        CX_ori.append(cx_ori)
        CY_ori.append(cy_ori)
    for arr in DD_ori:
        k1_ori=arr[0]
        k2_ori=arr[1]
        k5_ori=arr[4]
        K1_ori.append(k1_ori)
        K2_ori.append(k2_ori)
        K5_ori.append(k5_ori)



    # 从KK和DD中取出对应位置的数组
    FX=[]
    FY=[]
    CX=[]
    CY=[]
    K1=[]
    K2 = []
    K5 = []
    for arr in KK:
        fx = arr[0][0]
        fy = arr[1][1]
        cx = arr[0][2]
        cy = arr[1][2]
        FX.append(fx)
        FY.append(fy)
        CX.append(cx)
        CY.append(cy)
    for arr in DD:
        k1=arr[0]
        k2=arr[1]
        k5=arr[4]
        K1.append(k1)
        K2.append(k2)
        K5.append(k5)

    # # 数据
    # data = [FX, FY, CX, CY]
    # labels = ['FX', 'FY', 'CX', 'CY']
    #
    # # 创建箱线图
    # plt.boxplot(data, labels=labels)
    # plt.xlabel('Parameters')
    # plt.ylabel('Values')
    # plt.title('Boxplot of Parameters')
    # plt.yscale('log')  # 使用对数刻度
    # plt.show()

    # 计算平均值和标准差
    mean_FX_ori = np.mean(FX_ori)
    std_FX_ori = np.std(FX_ori)
    mean_FX = np.mean(FX)
    std_FX = np.std(FX)
    print(std_FX_ori,"------",FX_ori)
    print(std_FX,"------",FX)

    mean_FY_ori = np.mean(FY_ori)
    std_FY_ori = np.std(FY_ori)
    mean_FY = np.mean(FY)
    std_FY = np.std(FY)
    print(std_FY_ori,"------",FY_ori)
    print(std_FY,"------",FY)

    mean_CX_ori = np.mean(CX_ori)
    std_CX_ori = np.std(CX_ori)
    mean_CX = np.mean(CX)
    std_CX = np.std(CX)
    print(std_CX_ori,"------",CX_ori)
    print(std_CX,"------",CX)

    mean_CY_ori = np.mean(CY_ori)
    std_CY_ori = np.std(CY_ori)
    mean_CY = np.mean(CY)
    std_CY = np.std(CY)
    print(std_CY_ori,"------",CY_ori)
    print(std_CY,"------",CY)

    mean_K1_ori = np.mean(K1_ori)
    std_K1_ori= np.std(K1_ori)
    mean_K1 = np.mean(K1)
    std_K1 = np.std(K1)
    print(std_K1_ori,"------",K1_ori)
    print(std_K1,"------",K1)

    mean_K2_ori = np.mean(K2_ori)
    std_K2_ori = np.std(K2_ori)
    mean_K2 = np.mean(K2)
    std_K2 = np.std(K2)
    print(std_K2_ori,"------",K2_ori)
    print(std_K2,"------",K2)


    mean_K5_ori = np.mean(K5_ori)
    std_K5_ori = np.std(K5_ori)
    mean_K5 = np.mean(K5)
    std_K5 = np.std(K5)
    print(std_K5_ori,"------",K5_ori)
    print(std_K5,"------",K5)











