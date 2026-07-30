import cv2

# 1. 建立视频读取对象 (把路径换成你真实的 mp4 地址)
video_path = r'F:\机器学习和计算机视觉\大创\Tennis-master\data\videos\V010.mp4'
cap = cv2.VideoCapture(video_path)

# 2. 检查是否打开成功
if not cap.isOpened():
    print("打不开视频，请检查路径或编码格式！")
    exit()

while True:
    # 3. 逐帧读取：ret 是布尔值（是否读到图），frame 是图像矩阵
    ret, frame = cap.read()

    # 如果读不到帧（视频结束了），就跳出循环
    if not ret:
        break

    # 4. 显示画面
    cv2.imshow('Video Test', frame)

    # 5. 退出机制：按键盘上的 'q' 键退出
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 6. 释放资源并关闭窗口
cap.release()
cv2.destroyAllWindows()