# Import opencv

import cv2

# Loading cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Creating a function

def detect(gray,frame):
    # detectMultiscale(img, scaling , zones required to be accepted)
    #returns 2 tuples  
    faces = face_cascade.detectMultiScale(gray, 1.3 , 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_frame = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1 , 22)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(255,0,120),2)
        smiles = smile_cascade.detectMultiScale(gray,1.7,22)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(frame,(sx,sy),(sx+sw,sy+sh),(0,255,255),2)
    return frame
        
# Face detection using webcam (0) external cam (1)
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()