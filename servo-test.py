import maestro
import time
servo = maestro.Controller()
servo.setAccel(0,4)      #set servo 0 acceleration to 4
servo.setSpeed(0,40)     #set speed of servo 1
servo.setAccel(1,4)      #set servo 0 acceleration to 4
servo.setSpeed(1,40)     #set speed of servo 1
x = servo.getPosition(1) #get the current position of servo 1

servo.setTarget(0,1)  #set servo to move to center position
servo.setTarget(1,5000)  #set servo to move to center position

D = 2
time.sleep(D)

servo.setTarget(0,9999)  #set servo to move to center position
servo.setTarget(1,5000)  #set servo to move to center position

time.sleep(D)

servo.setTarget(0,9999)  #set servo to move to center position
servo.setTarget(1,9999)  #set servo to move to center position

time.sleep(D)

servo.setTarget(0,1)  #set servo to move to center position
servo.setTarget(1,9999)  #set servo to move to center position

time.sleep(D)

servo.setTarget(0,1)  #set servo to move to center position
servo.setTarget(1,5000)  #set servo to move to center position

servo.close()
