import cv2
import numpy as np

# we grab the capture as video first
cap = cv2.VideoCapture("example.mp4")

while cap.isOpened():

    # ret controls that whether the read function keep sending captures to the image object or not
    # if it sends, we can see the image object properly
    ret, image = cap.read()

    if ret:
        image = cv2.resize(image, (500, 500))

        # we convert bgr(rgb) color to hsv color palette
        # because hsv provide us vibrancy and brightness which means closer to human seeing than rgb model does
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # lower and upper limits of hue, saturation, value rates for red color
        lower = np.array([130, 100, 100])
        upper = np.array([179, 255, 255])

        # mask is used for distinguishing red(or which color we decide) object.
        # How it do that is with using limit rates it makes out of limits black and makes what we want white
        mask = cv2.inRange(hsv, lower, upper)

        # this section provide us to decide how far our frame from the object practically. I do not understand teorically well
        kernel = np.ones((1, 1), np.uint8)
        mask = cv2.erode(mask, kernel)


        # findcontours function returns an array containing contour points and we get this array whit 'contours' variable
        # also hie(hierarchy) means if you have nested objects on your mask hie orders them outer and inner.
        contours, hie = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # looping our contours array
        for c in contours:
            font = cv2.FONT_HERSHEY_SIMPLEX
            area = cv2.contourArea(c)
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)

            # moments function gets the object pixels which we want to detect in every second, on a video file or live camera
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # to ignore smaller red areas than we are looking for, we can use area variable
            if area > 400:
                # in drawContours function we use approx list instead 'contours' variable
                # third parameter is index of counter and we use -1 that means all contours we chose
                cv2.drawContours(image, [approx], -1, (255, 80, 80), 2)

                # methods for signing center pixel points of the object
                cv2.circle(image, (cX, cY), 5, (255, 80, 80), -1)
                cv2.putText(image, f"C{cX},{cY}", (cX - 30, cY + 20), font, 0.4, (255, 80, 80), 1, cv2.LINE_AA)

        cv2.imshow("image", image)
        cv2.imshow("mask", mask)
        k = cv2.waitKey(1)
        if (k == ord("q")) | (k == 27):
            break

    else:
        break


cap.release()
cv2.destroyAllWindows()

