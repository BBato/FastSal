import torch
import torchvision
import cv2
import numpy as np
import time, threading
import onnxruntime
import imutils
import maestro
import math
import multiprocessing
from simple_pid import PID
from PIL import Image
from generate_img import post_process_png
from utils import load_weight
from dataset.utils import pytorch_normalze
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import argparse

global servo
servo = maestro.Controller()
servo.setAccel(0,100)      #set servo 0 acceleration to 4
servo.setSpeed(0,100)     #set speed of servo 1
servo.setAccel(1,100)      #set servo 0 acceleration to 4
servo.setSpeed(1,100)     #set speed of servo 1

camera_image_size = (128, 96)
saveFrames = False
captureTime = 0
captureInterval = 1
initialCaptureTime = int(time.time())
last_reset = time.time()
useOtsu = False
shutter_speed = 2000

class PiVideoStream:
    def __init__(self, resolution=(320, 240), framerate=32):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.camera.shutter_speed = shutter_speed
        self.camera.iso = 800
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
        self.lastTime = time.time()
        self.ignoreFrames = 0
        servo.setTarget(self.channel, self.current_position)

    def updateError(self, newError):
        self.lastError = newError
        delta = -self.controller(self.lastError)
        self.current_position = self.current_position + delta
        self.current_position = max(self.current_position, self.servo_range[0])
        self.current_position = min(self.current_position, self.servo_range[1])
        servo.setTarget(self.channel, int(self.current_position))

    def reset(self):
        print("Reseting")
        self.lastError = int((self.initial_position - self.current_position)*0.1)

    def isCentered(self):
        if(self.current_position < self.initial_position + 800 and self.current_position > self.initial_position - 800):
            return True
        else:
            return False


    def start(self):
        return self


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Configuration for robotic device')
    parser.add_argument('-save', action='store', dest='save',
                        help='Save frames to output folder at regular intervals', default=False, type=bool)
    parser.add_argument('-interval', action='store', dest='interval',
                        help='Specify interval to save output frames', default=1, type=int)
    parser.add_argument('-otsu', action='store', dest='otsu',
                        help='Run the experiment using Otsu thresholding technique')
    parser.add_argument('-output_path', action='store', dest='output_path',
                        help='Path to output captured frames', default='./robot-test/')
    parser.add_argument('-visualize', action='store', dest='visualize',
                        help='Show captured frame and saliency map', default=False, type=bool)
    parser.add_argument('-pitch_Kp', action='store', dest='pitch_Kp',
                        help='PID controller: specify pitch K_p', default=3.5, type=float)
    parser.add_argument('-pitch_Ki', action='store', dest='pitch_Ki',
                        help='PID controller: specify pitch K_i', default=0.35, type=float)
    parser.add_argument('-pitch_Kd', action='store', dest='pitch_Kd',
                        help='PID controller: specify pitch K_d', default=0.0875, type=float)
    parser.add_argument('-yaw_Kp', action='store', dest='yaw_Kp',
                        help='PID controller: specify yaw K_p', default=1.4, type=float)
    parser.add_argument('-yaw_Ki', action='store', dest='yaw_Ki',
                        help='PID controller: specify yaw K_i', default=0.6, type=float)
    parser.add_argument('-yaw_Kd', action='store', dest='yaw_Kd',
                        help='PID controller: specify yaw K_d', default=0.15, type=float)
    parser.add_argument('-shutter_speed', action='store', dest='shutter_speed',
                        help='Select camera shutter time in nanoseconds', default=2000, type=int)
    args = parser.parse_args()

    captureInterval = args.interval
    saveFrames = args.save
    useOtsu = args.otsu
    visualize = args.visualize
    output_path = args.output_path
    shutter_speed = args.shutter_speed

    if(visualize):
        placeholder = np.zeros(shape=(128,96,1)).astype('uint8')
        cv2.imshow('original',placeholder)
        cv2.imshow('contours',placeholder)
        cv2.moveWindow('original',0,0)
        cv2.moveWindow('contours',130,0)

    sess_options = onnxruntime.SessionOptions()

    # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # Use OpenMP optimizations.
    sess_options.intra_op_num_threads = multiprocessing.cpu_count()

    ort_session = onnxruntime.InferenceSession("onnx/cocoA/fastsal_96_128.onnx", sess_options)

    vs = PiVideoStream().start()
    time.sleep(2.0)

    yaw = ServoController(4, PID(Kp=args.yaw_Kp, Ki=args.yaw_Ki, Kd=args.yaw_Kd , setpoint=0), (-100, 100), [1,9999] , 6000).start()
    pitch = ServoController(5, PID(Kp=args.pitch_Kp, Ki=args.pitch_Ki, Kd=args.pitch_Kd, setpoint=0), (-100, 100), [4800, 5700] , 5500).start()


    while True:

        # Capture the video frame by frame
        raw_camera_input = vs.read()
        raw_camera_input = cv2.flip(raw_camera_input,-1)

        # Convert to PIL object
        cv2_im = cv2.cvtColor(raw_camera_input, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        pil_im = pil_im.convert("RGB")
        input_image, _ = convert_vgg_img(pil_im, (96, 128))
        camera_input = input_image[np.newaxis, :, :, :]

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(camera_input)}
        ort_outs = ort_session.run(None, ort_inputs)
        saliency_output = ort_outs[0]

        # Prepare for display
        y = torch.nn.Sigmoid()(torch.tensor(saliency_output))
        y = y.detach().numpy()

        for i, prediction in enumerate(y[:, 0, :, :]):

            prediction = post_process_png(prediction, camera_image_size)
            prediction = np.repeat(prediction[:, :, np.newaxis], 3, axis=2)

            small_original = cv2.resize(raw_camera_input, (128,96))
            img_uint8 = np.uint8(prediction*255)

            # converting image into grayscale image 
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY) 
            
            # setting threshold of gray image 
            threshold = None
            if useOtsu:
                ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            else:
                ret, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours( 
                threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

            if len(contours)>0 and len(contours[0])>0: 

                pointsX = contours[0][:,0][:,0]
                pointsY = contours[0][:,0][:,1]
                    
                cX = int(sum(pointsX)/len(pointsX))
                cY = int(sum(pointsY)/len(pointsY))

                errorX = int(cX - camera_image_size[0]/2)
                errorY = int(camera_image_size[1]/2 - cY)

                yaw.updateError(errorX)
                pitch.updateError(errorY)

                if(visualize):
                    colorscale = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2BGR) 
                    contourImage = cv2.drawContours(colorscale, contours, 0, (0,128,255), 3)

                    if(len(contours)>0):
                        contourImage = cv2.drawContours(contourImage, contours, 0, (0,128,255), 3)
                    if(len(contours)>1):
                        contourImage = cv2.drawContours(contourImage, contours, 1, (255,128,0), 3)
                    if(len(contours)>2):
                        contourImage = cv2.drawContours(contourImage, contours, 2, (255,0,128), 3)

                    contourImage = cv2.circle(contourImage, (int(camera_image_size[0]/2), int(camera_image_size[1]/2)), 1, (0,255,0), 5)
                    contourImage = cv2.circle(contourImage, (cX, cY), 1, (255,0,0), 5)
                    cv2.imshow('contours', contourImage)
                    cv2.imshow('original', raw_camera_input)

                if(saveFrames):
                    if time.time() > initialCaptureTime + captureTime + captureInterval:
                        if(captureTime>=471):
                            exit()
                        cv2.imwrite(output_path+'t'+str(captureTime)+'.png', raw_camera_input)
                        captureTime += captureInterval

            else:
                contourImage = cv2.cvtColor(threshold.copy(), cv2.COLOR_GRAY2BGR) 
                contourImage = cv2.circle(contourImage, (int(camera_image_size[0]/2), int(camera_image_size[1]/2)), 1, (0,255,0), 5)
                cv2.imshow('contours', contourImage)

            cv2.waitKey(1)

