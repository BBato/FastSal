
import cv2


if __name__ == "__main__":

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame by frame
        ret, original_frame = vid.read()
        original_frame = cv2.flip(original_frame,-1)

        # Resize to standard size
        frame = cv2.resize(original_frame, (320, 240), interpolation=cv2.INTER_AREA)
        cv2.imshow("original", original_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()

    # Destroy all the windows
    cv2.destroyAllWindows()