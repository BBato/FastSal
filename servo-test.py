import maestro
import time
servo = maestro.Controller()
servo.setAccel(4,4)      #set servo 0 acceleration to 4
servo.setSpeed(4,40)     #set speed of servo 1
servo.setAccel(5,4)      #set servo 0 acceleration to 4
servo.setSpeed(5,40)     #set speed of servo 1
x = servo.getPosition(1) #get the current position of servo 1

servo.setTarget(4,1)  #set servo to move to center position
servo.setTarget(5,5000)  #set servo to move to center position

D = 2
time.sleep(D)

servo.setTarget(4,9999)  #set servo to move to center position
servo.setTarget(5,5000)  #set servo to move to center position

time.sleep(D)

servo.setTarget(4,9999)  #set servo to move to center position
servo.setTarget(5,9999)  #set servo to move to center position

time.sleep(D)

servo.setTarget(4,1)  #set servo to move to center position
servo.setTarget(5,9999)  #set servo to move to center position

time.sleep(D)

servo.setTarget(4,1)  #set servo to move to center position
servo.setTarget(5,5000)  #set servo to move to center position

servo.close()
