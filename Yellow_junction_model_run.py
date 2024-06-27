from ultralytics import YOLO
import cv2
import torch
from sort.sort import *
from functions import get_car, read_license_plate,write_csv1,is_stationary,write_csv
import numpy as np
import datetime

current_date = datetime.date.today()
current_time = datetime.datetime.now().time()

day_of_week = current_date.weekday()

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_name = days[day_of_week] 

print("Current date:", current_date)
print("Day of the week:", day_name)

results = {}
penalties=[]
track_ids=[]
challan_list={}
ids_to_lp={}
old_car_loc_his={}
marked_ids=[]
clear=True
hl1=10000
hl2=-10000
cam_id=1
tolerance=5
cnt_tolerance=4
var1=torch.FloatTensor([1.0,2.0,3.0]).cuda()

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8x.pt')


license_plate_detector = YOLO(r'Licence_plate_detector_weights\best.pt')


cap = cv2.VideoCapture(r"demo_videos\v1.mp4")


vehicles = [2, 3, 5, 7]
ret, frame = cap.read()
frame_nmr = -1
ret = True  


while ret:
    frame_nmr += 1
    temp_track_ids=track_ids
    # if frame_nmr>=100:
    #     break
    
    ret, frame = cap.read()
    # frame = torch.tensor(frame).to('cuda')
    if ret==False:
        break
    rows, cols, _ = frame.shape
    frame = frame[100:rows-10, 10:cols-10]
    frame=cv2.resize(frame,(1920,1080))
    # if frame_nmr<100:
    #     continue
    if frame_nmr%6!=0:
        continue
    
    # (650,210)
    # (1350,210)
    # (1580,730)
    # (480,730)
    print(track_ids)
    
    yellow_box = np.array([[687, 236], [1295, 236], 
                [1510, 610], [530, 610],[687, 236]],
               np.int32)
    
    
    # yellow_box = np.array([[780, 620], [1180, 620], 
    #             [1350, 820], [630, 820],[780, 620]],
    #            np.int32)
    
    lane1 = np.array([[754, 100], [870, 100], 
                [780, 613], [534, 613],[754, 100]],
               np.int32)
    lane2 = np.array([[870, 100], [990, 100], 
                [1032, 613], [780, 613],[870, 100]],
               np.int32)
 
    yellow_box= yellow_box.reshape((-1, 1, 2))
    lane1= lane1.reshape((-1, 1, 2))
    lane2= lane2.reshape((-1, 1, 2))
    
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        # cv2.circle(frame,(1900,1050),5,(255,255,0),5)
        cv2.putText(frame,str(frame_nmr),(50,50),cv2.FONT_HERSHEY_SIMPLEX ,2,(255,255,0),2)
        vis=False
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            
            
            if int(class_id) in vehicles:
                vis=True
                detections_.append([x1, y1, x2, y2, score])
                cx=int((x1+x2)/2)
                cy=int((y1+y2)/2)
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),500)
                result = cv2.pointPolygonTest(np.array(yellow_box, np.int32), (int(cx), int(cy)), False)
                if result>=0:
                    pass
                    # cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                
                
                

        # # track vehicles
        
        if vis:
            track_ids = mot_tracker.update(np.asarray(detections_))
        # print("track_id",track_ids)
        thl1=10000
        thl2=10000
        new_car_loc_his={}
        for ti in track_ids:
            a,b,c,d,iid=ti
            
            cv2.putText(frame,str(int(iid)),(int(a),int(b)),cv2.FONT_HERSHEY_SIMPLEX ,2,(0,255,0),2)
            cen1=(a+c)/2
            cen2=(b+d)/2
            
            in_lane1 = cv2.pointPolygonTest(np.array(lane1, np.int32), (int(cen1), int(cen2)), False)
            in_lane2  = cv2.pointPolygonTest(np.array(lane2, np.int32), (int(cen1), int(cen2)), False)
            
            if in_lane1>=0 and cen2<thl1:
                thl1=cen2
            
            elif in_lane2>=0 and cen2<thl2:
                thl2=cen2
            # FIND PREV POSITION FROM TEMP_TRACK IDS
            
            result = cv2.pointPolygonTest(np.array(yellow_box, np.int32), (int(cen1), int(cen2)), False)
            
            if result>=0:
                cv2.circle(frame,(int(cen1), int(cen2)),5,(0,0,255),-1)
                
                
                if iid not in marked_ids:
                    marked_ids.append(iid)
                    # write logic for implementing the required steps for
                    if hl1<cen2 and hl2<cen2 :
                        if iid not in penalties:
                            penalties.append(iid)
                            current_time = datetime.datetime.now().time()
                            challan_list[iid]={"device_id":cam_id,
                                                    "date":current_date,
                                                    "day":day_name,
                                                    "time":current_time,
                                                    "license_plate_number":"",
                                                    "confidence":0
                                                    }

                
                if iid in old_car_loc_his:
                    new_car_loc_his[iid]=old_car_loc_his[iid]
                    prev_center=old_car_loc_his[iid]["center"]
                    if is_stationary((cen1,cen2),prev_center,tolerance):
                        new_car_loc_his[iid]["cnt"]+=1
                        if new_car_loc_his[iid]["cnt"]>cnt_tolerance and iid not in penalties:
                            penalties.append(iid)
                            # print("llllllllllllllll",penalties)
                            current_time = datetime.datetime.now().time()
                            challan_list[iid]={"device_id":cam_id,
                                                "date":current_date,
                                                "day":day_name,
                                                "time":current_time,
                                                "license_plate_number":"",
                                                "confidence":0
                                                }

                            if iid in ids_to_lp and challan_list[iid]["confidence"]<ids_to_lp[iid]["confidence"]:
                                challan_list[iid]["license_plate_number"]=ids_to_lp[iid]["license_plate_text"]
                                challan_list[iid]["confidence"]=ids_to_lp[iid]["confidence"]
                    
                    else:
                        new_car_loc_his[iid]["cnt"]=1
                        new_car_loc_his[iid]["center"]=(cen1,cen2)
                else:
                    new_car_loc_his[iid]={"center":(cen1,cen2),"cnt":1}
            
            if iid in penalties:
                if iid in ids_to_lp and challan_list[iid]["confidence"]<ids_to_lp[iid]["confidence"]:
                                challan_list[iid]["license_plate_number"]=ids_to_lp[iid]["license_plate_text"]
                                challan_list[iid]["confidence"]=ids_to_lp[iid]["confidence"]
                                
        hl1=thl1
        hl2=thl2
        print("old--",old_car_loc_his)
        print("new--",new_car_loc_his)
        print(challan_list)
        old_car_loc_his=new_car_loc_his
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                cv2.imshow("a",license_plate_crop)
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                cv2.imshow("b",license_plate_crop_gray)
                
                
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                cv2.imshow("c",license_plate_crop_thresh)
                

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)
                
                
                
                
                if license_plate_text is not None:
                    if car_id not in ids_to_lp or license_plate_text_score>ids_to_lp[car_id]["confidence"]:
                        ids_to_lp[car_id]={
                        "license_plate_text":license_plate_text,
                        "confidence":license_plate_text_score,
                        }

                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
    cv2.polylines(frame,[yellow_box],isClosed=False,color=(255,255,0),thickness=2)
    cv2.polylines(frame,[lane1],isClosed=False,color=(255,0,255),thickness=2)
    cv2.polylines(frame,[lane2],isClosed=False,color=(0,0,255),thickness=2)
    
    cv2.imshow("Y.J.A.C. MODEL",frame)
    key = cv2.waitKey(33)
    if key == 27:
        break
    

# write results
print(results)
write_csv(results, './output.csv')
write_csv1(challan_list, './challan_list.csv')