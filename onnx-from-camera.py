from utils import load_weight
from dataset.utils import pytorch_normalze
import torch
import torchvision
import cv2
import numpy as np
from PIL import Image
from generate_img import post_process_png
import time, threading
import onnxruntime
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
from imutils.video import FPS
import imutils
import maestro

servo = maestro.Controller()
servo.setAccel(0,10)      #set servo 0 acceleration to 4
servo.setSpeed(0,100)     #set speed of servo 1
servo.setAccel(1,10)      #set servo 0 acceleration to 4
servo.setSpeed(1,100)     #set speed of servo 1

servoRangeYaw = [1,9999]
servoRangePitch = [5000, 9999]

servo_channel_pitch = 1
servo_channel_yaw = 0

global servoPitch
global servoYaw
servoPitch = 7500
servoYaw = 5000


def adjustCam(errorX, errorY):
    global servoYaw
    global servoPitch
    print("errorX: "+str(errorX))
    print("errorY: "+str(errorY))

    servoYaw = servoYaw + errorY
    servoPitch = servoPitch + errorX

    if(servoYaw>servoRangeYaw[1]):
        servoYaw = servoRangeYaw[1]

    if(servoYaw<servoRangeYaw[0]):
        servoYaw = servoRangeYaw[0]

    if(servoPitch>servoRangePitch[1]):
        servoPitch = servoRangePitch[1]

    if(servoPitch<servoRangePitch[0]):
        servoPitch = servoRangePitch[0]

    servo.setTarget(servo_channel_pitch, servoPitch)
    servo.setTarget(servo_channel_yaw, servoYaw)



class setInterval :
    def __init__(self,interval,action) :
        self.interval=interval
        self.action=action
        self.stopEvent=threading.Event()
        thread=threading.Thread(target=self.__setInterval)
        thread.start()

    def __setInterval(self) :
        nextTime=time.time()+self.interval
        while not self.stopEvent.wait(nextTime-time.time()) :
            nextTime+=self.interval
            self.action()

    def cancel(self) :
        self.stopEvent.set()

class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=32):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)
            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def convert_vgg_img(src, target_size):
    vgg_img = src
    original_size = vgg_img.size
    if isinstance(target_size, tuple) or isinstance(target_size, list):
        if target_size[0] != original_size[1] or target_size[1] != original_size[0]:
            vgg_img = vgg_img.resize((target_size[1], target_size[0]), Image.ANTIALIAS)
        elif isinstance(target_size, int):
            vgg_img = vgg_img.resize(
                (
                    int(original_size[0] / target_size),
                    int(original_size[2] / target_size),
                ),
                Image.ANTIALIAS,
            )
    vgg_img = np.asarray(vgg_img, dtype=np.float32)
    vgg_img = pytorch_normalze(torch.FloatTensor(vgg_img).permute(2, 0, 1) / 255.0)
    return vgg_img, np.asarray(original_size)

def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy()
        if tensor.requires_grad
        else tensor.cpu().numpy()
    )

def action():

    t1 = time.process_time()

    # Capture the video frame by frame
    original_frame = vs.read()
    original_frame = cv2.flip(original_frame,-1)

    # Resize to standard size
    frame = cv2.resize(original_frame, (320, 240), interpolation=cv2.INTER_AREA)

    # Convert to PIL object
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    
    pil_im = pil_im.convert("RGB")
    x, camera_image_size = convert_vgg_img(pil_im, (192, 256))
    x = x[np.newaxis, :, :, :]

    # Run model
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    y = ort_outs[0]



    # Prepare for display
    y = torch.nn.Sigmoid()(torch.tensor(y))

    y = y.detach().numpy()
    for i, prediction in enumerate(y[:, 0, :, :]):
        
        img_data = post_process_png(prediction, camera_image_size)
        result = np.where(img_data == np.amax(img_data))
        listOfCordinates = list(zip(result[0], result[1]))
        
        c_x = listOfCordinates[0][0]
        c_y = listOfCordinates[0][1]


        original_frame[listOfCordinates[0][0]][listOfCordinates[0][1]] = [255,0,0]
        original_frame[120][160] = [255,255,255]

        adjustCam(120 - c_x, c_y - 160)

        cv2.imshow("img_output_path", img_data)
        cv2.imshow("original", original_frame)
        
    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        # After the loop release the cap object
        vid.release()

        # Destroy all the windows
        cv2.destroyAllWindows()
        return

    t2 = time.process_time()
    interval = t2 - t1
    fps = 1 / interval
    print("FPS: " + str(fps))



if __name__ == "__main__":

    print("Loading model...")
    ort_session = onnxruntime.InferenceSession("onnx/cocoA/fastsal.onnx")

    # created a *threaded *video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from `picamera` module...")
    vs = PiVideoStream().start()
    time.sleep(2.0)
    fps = FPS()
    
    while True:
        fps.start()
        action()
        fps.stop()


    # # define a video capture object
    # vid = cv2.VideoCapture(0)

    # print("Scheduling interval")
    # # inter=setInterval(1,action)

    # while(True):
    #     action()


