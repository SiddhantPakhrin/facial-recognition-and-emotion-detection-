from cv2 import cv2
import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)


def insertOrUpdate(Id, Name, Age, Gender,Criminal_Records):
    conn = sqlite3.connect("faceinfo.db")
    cmd = "SELECT * FROM Users WHERE ID="+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist == 1):
        cmd="UPDATE Users SET Name='"+str(Name)+"'WHERE ID="+str(Id)
        cmd2="UPDATE Users SET Age='"+str(Age)+"'WHERE ID="+str(Id)
        cmd3="UPDATE Users SET Gender='"+str(Gender)+"'WHERE ID="+str(Id)
        cmd4="UPDATE Users SET Criminal_Records='"+str(Criminal_Records)+"'WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO Users(ID,Name,Age,Gender,Criminal_Records) Values("+str(Id)+",'"+str(Name)+"','"+str(Age)+"','"+str(Gender)+"','"+str(Criminal_Records)+"')"
        cmd2=""
        cmd3=""
        cmd4=""
    conn.execute(cmd)
    conn.execute(cmd2)
    conn.execute(cmd3)
    conn.execute(cmd4)

    conn.commit()
    conn.close()


Id=input('Enter User Id')
name=input('Enter User Name')
age=input('Enter User Age')
gen=input('Enter User Gender')
cr=input('Enter User Criminal Records')
insertOrUpdate(Id,name,age,gen,cr)
sampleNum = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5,)
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.imwrite("dataSet/User."+str(Id)+"." +
                    str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("face", img)
    cv2.waitKey(1)
    if(sampleNum > 100):
        break
    # if cv2.waitKey(1) == ord('q'):
    #     break


cam.release()
cv2.destroyAllWindows()
