import cv2

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    webcam_borders = cv2.Canny(frame_gray, 50, 150)
    #webcam_borders_blur = cv2.Canny(frame_blur, 100, 200)

    cv2.imshow('Webcam', frame)
    cv2.imshow('Canny Webcam', webcam_borders)
    #cv2.imshow('Canny Webcam Blur', webcam_borders_blur)

    if cv2.waitKey(1) == ord('v'):
        break

capture.release()
cv2.destroyAllWindows()
