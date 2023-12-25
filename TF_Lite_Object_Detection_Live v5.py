import os
import cv2
import numpy as np
import time
import pyrebase
from tensorflow.lite.python.interpreter import Interpreter

# Firebase Configuration
firebaseConfig={
    "apiKey": "AIzaSyAUM0Ae-kEaulkbWJyqud-MBPzTiqgr0OE",
    "authDomain": "smartswitch05-c375d.firebaseapp.com",
    "projectId": "smartswitch05-c375d",
    "storageBucket": "smartswitch05-c375d.appspot.com",
    "messagingSenderId": "643048685307",
    "appId": "1:643048685307:web:2cda2dd6837c871703ba5e",
    "measurementId": "G-0687DRB4TJ",
    "databaseURL": "https://smartswitch05-c375d-default-rtdb.firebaseio.com/"
    }

email="dery@sswitch.com"
password="madara001"

firebase=pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()
database = firebase.database()
auth=firebase.auth()
try:
    login=auth.sign_in_with_email_and_password(email, password)
    print("sucessfully login")
except:
    print("invalid credential")

state = 0

MODEL_NAME = "./detect.tflite"   #path file model
LABELMAP_NAME = "./labelmap.txt" #path file label map
min_conf_threshold = 0.5         #threshold deteksi
imW = 640   #cam width
imH = 480   #cam height

#load file label map
with open(LABELMAP_NAME, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
    
#load file model
interpreter = Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

#detail model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

outname = output_details[0]['name']
if ('StatefulPartitionedCall' in outname):  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

#inisialisasi perhitungan fps
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#inisialisasi kamera
cap = cv2.VideoCapture("rtsp://rtsp_c210:kopijahe@192.168.1.2:554/stream2")
#cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imH)
time.sleep(1)
ct = 0
period = 5
time_now = time.time()

while cap.isOpened():
    ct += 1
    t1 = cv2.getTickCount()      #mulai timer untuk kalkulasi fps
    ret = cap.grab()  #mengambil frame kamera
    if ct % 1 == 0:
        ret, frame = cap.retrieve()
        if not ret: break
        #time.sleep(0.2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        #fungsi normalize model jika menggunakan model floating
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
        person_count = 0
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin-7),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                if labels[int(classes[i])] == 'person':
                    person_count += 1
                    print(person_count)

            if person_count > 0:
                time_now = time.time()
                if state == 0:
                    database.child("Switch")
                    data = {"Node1": "1", "Node2": "1", "Node3": "1"}
                    database.update(data)
                    state = 1
                    
            else:
                if time.time() >= (time_now + period):
                    if state == 1:
                        database.child("Switch")
                        data = {"Node1": "0", "Node2": "0", "Node3": "0"}
                        database.update(data)
                        state = 0

        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Object detector', frame)

        #menghitung fps
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        if cv2.waitKey(1) == ord('q'):
            break

cv2.destroyAllWindows()
cap.release()

