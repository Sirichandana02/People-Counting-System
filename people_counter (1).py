import cv2
import datetime  #provides fun's to deal with date,time
import imutils   #used for basic img processing fun's
import numpy as np
from centroidtracker import CentroidTracker

protopath = "MobileNetSSD_deploy.prototxt"                                                            #obj detection model
modelpath = "MobileNetSSD_deploy.caffemodel"                                                          #ssd-->single shot detector
                                                                                                      #mobilenet---->classification and recognition
                                                                                                      #ssd---->framework to realize multibox detector
                                                                                                      #.prototext----->struct of neural network stored as ...(contains img classification and segmentation models)
                                                                                                      #.caffemodel file---->weights of layers of neural network stored as...
                                                                                                      #caffe model --->for more accurate Face_Reco_imgs detection


detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)               #to load/import our model
                                                                                            #dnn---->to work with any pre-trained model

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",                     #since it is a generic obj detection model...so we mention all the classes it has.
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


#initilize tracker
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
#maxDisappeared------->no.of frames our tracker waits for an obj when it moves out of frame



# Non Max Suppersion algorithm
def non_max_suppression_fast(boxes, overlapThresh):          #removes all the noises like redundant bounding boxes (marks same obj multiple times)
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))




def main():
    cap = cv2.VideoCapture(0)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1             #for moving each frame

        (H, W) = frame.shape[:2]  # to get height, width------>extracts rows,columns from shape tupple

        #detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)       #blob--->binary large obj........can read text, binary data..........detection and extraction from img

        detector.setInput(blob)
        person_detections = detector.forward()  # contains all the detection results from model

        rects = []                                   #list to save all cordinates
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")

                rects.append(person_box)       #append all cordinates of persons which our model is detecting

        boundingboxes = np.array(rects)                      #storing in array
        boundingboxes = boundingboxes.astype(int)            #converting into integers

        rects = non_max_suppression_fast(boundingboxes, 0.3)     #contains crt bounding boxes and does not has noises in it

        #passing bounding boxes , its cordinates to tracker
        objects = tracker.update(rects)     #---->gives bounding box and id to each person
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox            #cordinates of bbox
            x1 = int(x1)                     #cvt to integers
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)        #shows id in string format
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)     #text displayed just above bbox of person


            if objectId not in object_id_list:
                object_id_list.append(objectId)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)              #cvt fps value  to 2 decimal points

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)

        lpc_txt = "LPC: {}".format(lpc_count)
        opc_txt = "OPC: {}".format(opc_count)

        cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()