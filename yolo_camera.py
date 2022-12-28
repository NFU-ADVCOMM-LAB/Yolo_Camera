import cv2
import numpy
from yolo import yolov5_addon

def Camera(camera_id, frame_height, frame_width):
        camera = cv2.VideoCapture(camera_id)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, frame_height)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_width)
        if not camera.isOpened():
            print("Cannot open camera")
            exit()
        return camera

def main():
    yolo = yolov5_addon('./Yolov5_Test.pt') #[權重的存放路徑]
    cam = Camera(0, 1280, 720) #[相機ID(一般來說為0), 畫面高, 畫面寬]
    while True:
        ret, frame = cam.read() #從相機取得畫面
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        yolo_frame = yolo.pred(frame) # Yolo
        yolo_show_frame = numpy.array(yolo_frame)
        cv2.imshow('frame', yolo_show_frame) # 顯示畫面
        if cv2.waitKey(1) == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if (__name__== "__main__"):
    main()
