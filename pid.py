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
from multiprocessing import Pool,TimeoutError 
import multiprocessing
import math
from simple_pid import PID

servo = maestro.Controller()
servo.setAccel(0,6)      #set servo 0 acceleration to 4
servo.setSpeed(0,6)     #set speed of servo 1
servo.setAccel(1,6)      #set servo 0 acceleration to 4
servo.setSpeed(1,6)     #set speed of servo 1

servoRangeYaw = [1,9999]
servoRangePitch = [4500, 7000]

servo_channel_pitch = 5
servo_channel_yaw = 4

global servoPitch
global servoYaw
servoPitch = 7500
servoYaw = 5000

global pitchDelta1
global pitchDelta2
pitchDelta1 = 0
pitchDelta2 = 0

pid1 = PID(Kp=3, Ki=0.4, Kd=0.1, setpoint=0)
pid1.output_limits = (-1000, 1000)
pid2 = PID(Kp=5, Ki=0.1, Kd=0.01, setpoint=0)
pid2.output_limits = (-1000, 1000)

global start_time
global last_time
start_time = time.time()
last_time = start_time

def resetCamPosition():
    servo.setTarget(servo_channel_pitch, 5500)
    servo.setTarget(servo_channel_yaw, 5000)

def adjustCam(errorX, errorY):


    global servoYaw
    global servoPitch
    global start_time
    global last_time
    global pitchDelta1
    global pitchDelta2
    
    current_time = time.time()
    pitchDelta1 = -pid1(errorY)
    pitchDelta2 = -pid2(errorX)
    
    if(abs(pitchDelta1)>5):
        servoYaw = int(servoYaw+pitchDelta1)

    if(abs(pitchDelta2)>5):
        servoPitch = int(servoPitch+pitchDelta2)

    if(servoYaw>servoRangeYaw[1]):
        servoYaw = servoRangeYaw[1]

    if(servoYaw<servoRangeYaw[0]):
        servoYaw = servoRangeYaw[0]

    last_time = current_time

    if(servoPitch>servoRangePitch[1]):
         servoPitch = servoRangePitch[1]

    if(servoPitch<servoRangePitch[0]):
        servoPitch = servoRangePitch[0]

    servo.setTarget(servo_channel_pitch, servoPitch)
    servo.setTarget(servo_channel_yaw, servoYaw)



class KeyboardThread(threading.Thread):

    def __init__(self, input_cbk = None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            self.input_cbk(input()) #waits to get input + Return

def my_callback(inp):
    global servoYaw
    #evaluate the keyboard input
    a = inp.split(',')
    
    """ pid1.tunings = (float(a[0]), float(a[1]), float(a[2]))
    print(pid1.tunings) """

    servoYaw = servoYaw+int(a[0])
    servo.setTarget(servo_channel_yaw, servoYaw)

#start the Keyboard thread
kthread = KeyboardThread(my_callback)

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

# class FastSalRunner:

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

def onnx(x):
    
    # Run model
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    y = ort_outs[0]
    
    return y


def capture():

    # Capture the video frame by frame
    original_frame = vs.read()
    original_frame = cv2.flip(original_frame,-1)

    # cv2.imshow("original", original_frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     # After the loop release the cap object
    #     vid.release()

    #     # Destroy all the windows
    #     cv2.destroyAllWindows()
    #     return
    # return

    # Resize to standard size
    frame = original_frame#cv2.resize(original_frame, (320, 240), interpolation=cv2.INTER_AREA)

    # Convert to PIL object
    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    
    pil_im = pil_im.convert("RGB")
    camera_image_size = (128, 96)
    x, _ = convert_vgg_img(pil_im, (96, 128))
    x = x[np.newaxis, :, :, :]

    return x, camera_image_size, original_frame
    



def visualize(y, camera_image_size, original_frame):
    
    # Prepare for display
    y = torch.nn.Sigmoid()(torch.tensor(y))

    y = y.detach().numpy()

    for i, prediction in enumerate(y[:, 0, :, :]):

        prediction = post_process_png(prediction, camera_image_size)
        prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2)

        img_uint8 = np.uint8(prediction*255)

        # converting image into grayscale image 
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY) 
        
        # setting threshold of gray image 
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
        
        # using a findContours() function 
        contours, _ = cv2.findContours( 
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

        sumX = 0
        sumY = 0
        for c in contours:
            sumX += c[0][0][0]
            sumY += c[0][0][1]
        length = len(contours)
        cX = int(sumX/length)
        cY = int(sumY/length)

        prediction = cv2.circle(prediction, (cX, cY), 1, (255,0,0), 5)

        errorX = int(cX - camera_image_size[0]/2)
        errorY = int(camera_image_size[1]/2 - cY)
        center = cv2.circle(prediction, (int(camera_image_size[0]/2), int(camera_image_size[1]/2)), 1, (0,255,0), 5)
        #adjustCam(errorY, errorX )
        print('errorX',errorX)

        cv2.imshow("threshold", prediction)
        cv2.waitKey(1)

            
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # After the loop release the cap object
        vid.release()

        # Destroy all the windows
        cv2.destroyAllWindows()
        exit()


def loop(x, camera_image_size, original_frame):
    y = onnx(x)
    visualize(y, camera_image_size, original_frame)


if __name__ == "__main__":

    resetCamPosition()

    sess_options = onnxruntime.SessionOptions()
    # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
    sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    ort_session = onnxruntime.InferenceSession("onnx/cocoA/fastsal_96_128.onnx", sess_options)

    # created a *threaded *video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    vs = PiVideoStream().start()
    time.sleep(2.0)

    # period = 1
    # halfPeriod = period/2
    # with Pool(processes=2) as pool:

    #     x, camera_image_size, original_frame = capture()
    #     print('capture #1 ready')
    #     r2 = pool.apply_async(loop, (x, camera_image_size, original_frame))
    #     print('r2 started')
    #     time.sleep(halfPeriod)

    #     while True:
    
    #         # x, camera_image_size, original_frame = capture()
    #         # print('capture #2 ready')
    #         # r1 = pool.apply_async(loop, (x, camera_image_size, original_frame))
    #         # print('r1 started')
    #         # time.sleep(halfPeriod)

    #         r2.get()
    #         print('r2 finished')
    #         x, camera_image_size, original_frame = capture()
    #         print('capture #1 ready')
    #         r2 = pool.apply_async(loop, (x, camera_image_size, original_frame))
    #         print('r2 started')
    #         time.sleep(halfPeriod)

    #         # r1.get()



    while True:
        x, camera_image_size, original_frame = capture()
        loop(x, camera_image_size, original_frame)


        #showCameraFeed(original_frame)


def showCameraFeed(original_frame):
    cv2.imshow('original_frame',original_frame)

                    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        # After the loop release the cap object
        vid.release()

        # Destroy all the windows
        cv2.destroyAllWindows()
        exit()