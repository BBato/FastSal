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

global servo
servo = maestro.Controller()
servo.setAccel(0,100)      #set servo 0 acceleration to 4
servo.setSpeed(0,100)     #set speed of servo 1
servo.setAccel(1,100)      #set servo 0 acceleration to 4
servo.setSpeed(1,100)     #set speed of servo 1



global pX
global pY
pX = None
pY = None


def patrol():
    global patrolling
    global servoPitch
    global servoYaw
    servoPitch += int((initialPitch - servoPitch)*0.1)
    servoYaw += int((initialYaw - servoYaw)*0.1)
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
    #evaluate the keyboard input
    a = inp.split(',')
    pid1.tunings = (float(a[0]), float(a[1]), float(a[2]))
    print(pid1.tunings)

#start the Keyboard thread
kthread = KeyboardThread(my_callback)



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



class ServoController:
    def __init__(self, channel, controller, output_limits, servo_range, initial_position):
        self.controller = controller
        self.controller.output_limits = output_limits
        self.servo_range = servo_range
        self.channel = channel
        self.current_position = initial_position
        self.initial_position = initial_position
        self.lastError = 0
        self.lastTime = time.time_ns()
        servo.setTarget(self.channel, self.current_position)

    def updateError(self, newError):
        self.lastError = newError

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):

        while True:

            if(time.time_ns() > self.lastTime + 5):

                delta = -self.controller(self.lastError)
                #print(delta)

                #if(abs(delta)>5):
                self.current_position = self.current_position+delta
                #elif(abs(pitchDelta1)>2):
                #    servoYaw = int(servoYaw+pitchDelta1/abs(pitchDelta1))
                #print(delta)
                if(self.current_position>self.servo_range[1]):
                    self.current_position = self.servo_range[1]
                if(self.current_position<self.servo_range[0]):
                    self.current_position = self.servo_range[0]

                servo.setTarget(self.channel, int(self.current_position))
                self.lastTime = time.time_ns()

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
    global input_image

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
    input_image, _ = convert_vgg_img(pil_im, (96, 128))
    x = input_image[np.newaxis, :, :, :]

    return x, camera_image_size, original_frame
    



def visualize(y, camera_image_size, original_frame):
    global pX
    global pY

    # Prepare for display
    y = torch.nn.Sigmoid()(torch.tensor(y))

    y = y.detach().numpy()

    for i, prediction in enumerate(y[:, 0, :, :]):

        prediction = post_process_png(prediction, camera_image_size)
        prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2)

        small_original = cv2.resize(original_frame, (128,96))
        cv2.imshow('original', small_original)

        img_uint8 = np.uint8(prediction*255)

        # converting image into grayscale image 
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY) 
        
        # setting threshold of gray image 
        #ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        #kernel = np.ones((3,3), np.uint8)       
        #threshold = cv2.erode(threshold, kernel, iterations=5)

        # using a findContours() function 
        contours, _ = cv2.findContours( 
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 


        top = None
        bottom = None
        left = None
        right = None

        if len(contours)>0 and len(contours[0])>0: 

            for c in contours[0]:

                cnX = c[0][0]
                cnY = c[0][1]

                if top is None:
                    top = cnY
                    bottom = cnY
                    right = cnX
                    left = cnX
                    continue

                if(cnX > right):
                    right = cnX
                if(cnX < left):
                    left = cnX
                if(cnY > bottom):
                    bottom = cnY
                if(cnY < top):
                    top = cnY
            
            cX = int((left+right)/2)
            cY = int((top+bottom)/2)

            if pX is None:
                pX = int(camera_image_size[0]/2)
                pY = int(camera_image_size[1]/2)
            else:
                dX = cX - pX
                pX += int(dX *0.85)
                dY = cY - pY
                pY += int(dY *0.85)

            prediction = cv2.circle(prediction, (pX, pY), 1, (255,0,0), 5)

            errorX = int(pX - camera_image_size[0]/2)
            errorY = int(camera_image_size[1]/2 - pY)

            colorr = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2BGR) 
            contourImage = cv2.drawContours(colorr, contours, 0, (0,128,255), 3)

            if(len(contours)>0):
                contourImage = cv2.drawContours(contourImage, contours, 0, (0,128,255), 3)
            if(len(contours)>1):
                contourImage = cv2.drawContours(contourImage, contours, 1, (255,128,0), 3)
            if(len(contours)>2):
                contourImage = cv2.drawContours(contourImage, contours, 2, (255,0,128), 3)

            contourImage = cv2.circle(contourImage, (int(camera_image_size[0]/2), int(camera_image_size[1]/2)), 1, (0,255,0), 5)
            contourImage = cv2.circle(contourImage, (pX, pY), 1, (255,0,0), 5)
            cv2.imshow('contours', contourImage)

            center = cv2.circle(prediction, (int(camera_image_size[0]/2), int(camera_image_size[1]/2)), 1, (0,255,0), 5)
            yaw.updateError(errorX)
            pitch.updateError(errorY)

        else:
            contourImage = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2BGR) 
            contourImage = cv2.circle(contourImage, (int(camera_image_size[0]/2), int(camera_image_size[1]/2)), 1, (0,255,0), 5)
            cv2.imshow('contours', contourImage)
            #patrol()

        #cv2.imshow("threshold", prediction)
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

    test_img = np.zeros(shape=(128,96,1)).astype('uint8')
    cv2.imshow('original',test_img)
    cv2.imshow('contours',test_img)
    cv2.moveWindow('original',0,0)
    cv2.moveWindow('contours',130,0)

    sess_options = onnxruntime.SessionOptions()
    # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # Use OpenMP optimizations. Only useful for CPU, has little impact for GPUs.
    sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    ort_session = onnxruntime.InferenceSession("onnx/cocoA/fastsal_96_128.onnx", sess_options)

    # created a *threaded *video stream, allow the camera sensor to warmup,
    # and start the FPS counter
    vs = PiVideoStream().start()
    yaw = ServoController(4, PID(Kp=0.007, Ki=0.004, Kd=0.001 , setpoint=0), (-100, 100), [1,9999] , 5000).start()
    pitch = ServoController(5, PID(Kp=0.007, Ki=0.004, Kd=0.001 , setpoint=0), (-100, 100), [4500, 6000] , 5500).start()
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