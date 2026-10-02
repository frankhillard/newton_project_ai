import cv2
path = './data/videos/P112065_1.webm'
cap = cv2.VideoCapture(path)
print('opened:', cap.isOpened())
print('fps:', cap.get(cv2.CAP_PROP_FPS))
print('frame_count:', cap.get(cv2.CAP_PROP_FRAME_COUNT))
ret, frame = cap.read()
print('read first frame ok:', ret)
if ret:
    print('frame shape:', frame.shape)