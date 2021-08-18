# -*- coding:utf-8 -*-

'''
仅作为以圆形目标作为运动员的代替的演示实验
'''

import os
import socket
import time

import cv2 as cv
import cv2.aruco as aruco
import torch

from protobuf import data

loc_information = []
time_information = []
reference = 4000  # 预定义marker之间的实际间隔距离

# 修改主机时间
os.system('sudo date - s 00:00:00')


# marker位置
def marker_detect(gray):
    markers_result = dict()
    dictionary = aruco.Dictionary_create(20, 5)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is not None:
        ids = ids.reshape(-1)
        ids = ids.tolist()
        i = 0
        print(corners)
        for id in ids:
            Ord = corners[i].reshape(4, 2)
            x_ord = (abs(Ord[0][0] + Ord[2][0]) + abs(Ord[1][0] + Ord[3][0])) / 4
            markers_result.update({str(id): x_ord})
            i = i + 1
    else:
        markers_result = None
    return markers_result


# object位置获取
def obj_detect(gray):
    obj_x = []
    gray = cv.GaussianBlur(gray, (7, 7), 0)  # 高斯滤波
    # canny算法边缘检测（）
    edged = cv.Canny(gray, 80, 200)
    # 边缘的形态学变化
    edged = cv.dilate(edged, None, iterations=3)
    edged = cv.erode(edged, None, iterations=3)
    circles = cv.HoughCircles(
        edged, cv.HOUGH_GRADIENT, 1, 500, param1=100, param2=30, minRadius=0, maxRadius=60)
    if circles is not None:  # 如果识别出圆nvb
        for circle in circles[0]:
            #  获取圆的坐标与半径
            obj_x = int(circle[0])
            y = int(circle[1])
            r = int(circle[2])
            cv.circle(frame, (obj_x, y), r, (0, 0, 255), 3)  # 标记圆
            cv.circle(frame, (obj_x, y), 3, (255, 255, 0), -1)  # 标记圆心
    else:
        obj_x = 0
    # # 在边缘图像中寻找轮廓 （输入图像， 轮廓检索方式， 轮廓近似方式）
    # counts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # counts = imutils.grab_contours(counts)
    # for c in counts:
    #     if cv.contourArea(c) < 400:
    #         continue
    #     ep = 0.01 * cv.arcLength(c, True)
    #     approx = cv.approxPolyDP(c, ep, 1)  #
    #     if len(approx) < 6:
    #         continue
    #     mu = cv.moments(c)
    #     box = cv.minAreaRect(c)
    #     # 计算矩形的顶点坐标
    #     box = cv.boxPoints(box)
    #     box = np.array(box, dtype="int")
    #     obj_x = mu['m10'] / mu['m00']  # 轮廓contour的质心的横坐标
    return obj_x


def obj_swim_detect(img):
    torch.hub


# 解算实际距离
def real_distance(markers_result, obj_result, reference):
    ids = list(markers_result.keys())
    ids = list(map(int, ids))
    id_conuts = len(ids)
    ids.sort()
    obj_ord = obj_result
    print(id_conuts)
    if id_conuts < 2:
        print('Lack enough Markers! Need two markers at least!!!')
        real_dis = 0
    elif (obj_result is None) or obj_ord > markers_result[str(ids[-1])]:
        print('No object!!! Expecting in next UAV‘s information')
        real_dis = 0
    elif obj_ord > markers_result[str(ids[id_conuts - 2])]:
        K = abs(ids[id_conuts - 1] - ids[id_conuts - 2]) * reference / abs(
            markers_result[str(ids[id_conuts - 1])] - markers_result[str(ids[id_conuts - 2])])
        real_dis = K * abs(obj_ord - markers_result[str(ids[id_conuts - 2])]) + reference * ids[
            id_conuts - 2]
    elif (id_conuts - 3) >= 0 and obj_ord > markers_result[str(ids[id_conuts - 3])]:
        K = abs(ids[id_conuts - 2] - ids[id_conuts - 3]) * reference / abs(
            markers_result[str(ids[id_conuts - 2])] - markers_result[str(ids[id_conuts - 3])])
        real_dis = K * abs(obj_ord - markers_result[str(ids[id_conuts - 3])]) + reference * ids[
            id_conuts - 3]
    elif (id_conuts - 4) >= 0 and obj_ord > markers_result[str(ids[id_conuts - 4])]:
        K = abs(ids[id_conuts - 3] - ids[id_conuts - 4]) * reference / abs(
            markers_result[str(ids[id_conuts - 3])] - markers_result[str(ids[id_conuts - 4])])
        real_dis = K * abs(obj_ord - markers_result[str(ids[id_conuts - 4])]) + reference * ids[
            id_conuts - 4]
    else:
        print('No measured information')
        real_dis = 0
    return real_dis


uav = 1
# 获取USB_camera视频
video_src = cv.VideoCapture(1)
# 视频的保存
fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
# 视频保存路径
path = ''
out = cv.VideoWriter()
out.open(r"。/USBcamera.mp4",
         fourcc, 25, (1920, 1080))

uav_state = 1
if video_src.isOpened():
    print('video is opened')
    camera_state = 1
    # 设置视频分辨率和帧率
    video_src.set(cv.CAP_PROP_FRAME_WIDTH, 1080)
    video_src.set(cv.CAP_PROP_FRAME_HEIGHT, 960)
    video_src.set(cv.CAP_PROP_FPS, 25)
    Time = video_src.get(cv.CAP_PROP_POS_MSEC)
    milliseconds = video_src.get(cv.CAP_PROP_POS_MSEC)
else:
    print('video is failed to open!!')

ip_port = ('192.168.31.104', 8090)
# 建立socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 绑定ip/port
sock.connect(ip_port)

mav = data.MAVLink(1, uav, 1)
start = data.MAVLink_start_message(0)
start_buf = start.pack(mav)
sock.sendall(start_buf)
time.sleep(1)
while True:

    open_bool, frame = video_src.read()
    now_time = time.time()
    print(now_time)
    timestamp = round(now_time * 1000)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 检测 marker 位置
    markers_result = marker_detect(gray)
    print(markers_result)
    # 识别圆形轮廓，并提取形心
    obj_result = obj_detect(gray)
    # 计算实际位置
    if markers_result is not None:
        loc = real_distance(markers_result, obj_result, reference)
        # loc = round(loc, 2)
    else:
        loc = 0
    # 位置与时间戳信息传输 

    data_validity = 2
    font = cv.FONT_HERSHEY_SIMPLEX
    frame = cv.putText(frame, str(timestamp), (10, 50), font, 1,
                       (0, 255, 255), 2, cv.LINE_AA)
    frame = cv.putText(frame, str(loc), (10, 100), font, 1,
                       (0, 255, 255), 2, cv.LINE_AA)
    i_result = [timestamp, loc]

    message = data.MAVLink_measurment_info_message(timestamp, float(loc), camera_state, uav_state,
                                                   data_validity)
    buf = message.pack(mav)
    sock.sendall(buf)
    out.write(frame)
    cv.imshow('result', frame)
    if cv.waitKey(10) & 0xFF == 27:
        break
video_src.release()
cv.destroyAllWindows()
# np.savetxt('./result.txt', result)
