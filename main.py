from flask import Flask, Response, render_template
from ultralytics import YOLO
import supervision as sv
import numpy as np
import cv2

app = Flask(__name__) # declare the app

camera = cv2.VideoCapture(1) # declare the camera

# create detection zone
ZONE_POLYGON = np.array([
  [0,344//2],
  [1358//2,344//2],
  [1358//2,488//2],
  [0,488//2],
  [0,344//2]
])

def gen_frames():
  model = YOLO("yolov8n.pt")

  # set bbox
  box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
  )

  # set zone on frame
  zone = sv.PolygonZone(
    polygon=ZONE_POLYGON,
    frame_resolution_wh=tuple([640,480])
  )

  zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone,
    color=sv.Color.red(),
    thickness=2,
    text_thickness=4,
    text_scale=2
  )

  while True:
    success, frame = camera.read() # read the camera frame

    if not success:
      break
    else:
      result = model(frame, agnostic_nms=True)[0] # predict objects in frame; set the result uncrashed to other objects
      detections = sv.Detections.from_yolov8(result) # detect all objects
      detection = detections[detections.class_id==0] # filter object only person
      
      # set bbox labels
      labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _,_, confidence, class_id, _
        in detection
      ]

      # declare the frame with box annotator
      frame = box_annotator.annotate(
        scene=frame,
        detections=detections,
        labels=labels
      )

      zone.trigger(detections=detections) # activate detection only in the zone
      frame = zone_annotator.annotate(scene=frame) # add annotation in the frame
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