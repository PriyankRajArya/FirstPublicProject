import cv2


#Load the pre-trained haar cascode dectector

face_casecade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#Open webcam 
cap = cv2.VideoCapture(0)                      #0 = default camera

while True:
    ret,frame=cap.read()
    if not ret:
        break

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect Faces
    faces=face_casecade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    #Draw a rectangle 
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    #Show the Result 
    cv2.imshow('Face Detection',frame)

    #Press 'q to end the Camera
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



