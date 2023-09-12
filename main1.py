from flask import Flask, Response, render_template
from ultralytics import YOLO
# import supervision as sv
import numpy as np
import cv2

app = Flask(__name__) # declare the app

import pandas as pd
from tracker import *

model=YOLO('yolov8n.pt') # pre-trained model --> s = small

# get coordinate of cursor
# def RGB(event, x, y, flags, param):
#   if event == cv2.EVENT_MOUSEMOVE:
#     colorsBGR = [x, y]
#     print(colorsBGR)

# # window behaviour
# cv2.namedWindow('RGB')
# cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(1)

def gen_frames():
  # access file object class list
  my_file = open("coco.txt", "r")
  data = my_file.read()
  class_list = data.split("\n")
  #print(class_list)

  tracker = Tracker() # define tracker from modul tracker.py

  # line point (y)
  cy1 = 326
  cy2 = 330
  offset = 3 # object detection radius from line

  vh_down = {}
  vh_up = {}

  counter_down = []
  counter_up = []

  while True:
    _,frame = cap.read() # define window video frame

    frame = cv2.resize(frame,(1020,500)) # size frame
    results = model.predict(frame) # prediction results of frame
    # print(results):

    a = results[0].boxes.data # list of predictions
    # print(a)
    px = pd.DataFrame(a).astype("float") # arrange prediction object to table
    # print(px)

    list=[] # container for detected car in object id

    for index,row in px.iterrows():
      # print(row)
      y1 = int(row[1]) # top
      x1 = int(row[0]) # left
      x2 = int(row[2]) # right
      y2 = int(row[3]) # bottom

      d = int(row[5])
      c = class_list[d] # object class

      if 'car' in c: # filter object -- only car
        list.append([x1,y1,x2,y2]) # add car to container

      bbox_id = tracker.update(list) # assign id and position object (car)
      for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox # get coordinate and id
        cx = int(x3+x4)//2 # get x coordinate center
        cy = int(y3+y4)//2 # get y coordinate center

        # count object going down
        if cy < (cy1+offset) and cy > (cy1-offset):
          vh_down[id] = cy
        if id in vh_down:
          if cy < (cy2+offset) and cy > (cy2-offset):
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1) # object dot
            cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # show object id
            if id not in counter_down and id not in counter_up:
              counter_down.append(id)

        # count object going up
        if cy < (cy2+offset) and cy > (cy2-offset):
          vh_up[id] = cy
        if id in vh_up:
          if cy < (cy1+offset) and cy > (cy1-offset):
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1) # object dot
            cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # show object id
            if id not in counter_up and id not in counter_down:
              counter_up.append(id)

    # set line (threshold)
    cv2.line(frame,(274,cy1),(814,cy1),(255,255,255),1) # line 1
    cv2.putText(frame,('line 1'),(274,318),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # show line label
    cv2.line(frame,(177,cy2),(927,cy2),(255,255,255),1)
    cv2.putText(frame,('line 2'),(181,363),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # show line label

    # print(vh_down)
    # print(counter_down)
    # print(len(counter))

    total_objects_down = len(counter_down)
    cv2.putText(frame,('going down: ')+str(total_objects_down),(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # show sum of counting
    total_objects_up = len(counter_up)
    cv2.putText(frame,('going up: ')+str(total_objects_up),(60,70),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2) # show sum of counting

    ret, buffer = cv2.imencode('.jpg',frame) # encode the frame
    frame = buffer.tobytes() # save the frame into data bytes

    # send the frame as a HTTP streaming response
    yield (
      b'--frame\r\n'
      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
    )

@app.route('/')
def index():
  return 'Index Page'

@app.route('/world')
def world():
  return 'Hello World'

@app.route('/hello/<name>')
def hello(name='Visitor'):
  return render_template('index.html', name=name) # combine python into html templates with parameters 'name'

@app.route('/video_feed')
# live streaming --> update frame countinously
def video_feed():
  return Response( # send data video
    gen_frames(), # dynamical frames
    # header: live streaming content type; split by frame
    mimetype='multipart/x-mixed-replace; boundary=frame'
  )