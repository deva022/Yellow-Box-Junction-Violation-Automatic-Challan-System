import string
import easyocr
import math

reader = easyocr.Reader(['en'], gpu=False)

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

def is_stationary(point, center, radius):
    x, y = point
    cx, cy = center
    distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance <= radius



def write_csv1(results, output_path):
    
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{}\n'.format('Device_id', 'Date', 'Day',
                                                'Time', 'Licence_Plate_Number', 
                                                'Confidence'))
        
        
        
        for iids,ids in results.items():
            f.write('{},{},{},{},{},{}\n'.format(ids["device_id"],
                      ids["date"],                            
                      ids["day"],
                      ids["time"],                            
                      ids["license_plate_number"],                            
                      ids["confidence"], 
                    ))
        f.close()
    


def write_csv(results, output_path):
   
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))
        
        
        
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                # print("aaa",results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    
    if len(text) != 10:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in dict_char_to_int.keys()) and \
       (text[8] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[8] in dict_char_to_int.keys()) and \
       (text[9] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[9] in dict_char_to_int.keys()) and \
       (text[7] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[7] in dict_char_to_int.keys()):
        return True
    else:
        return False


def format_license(text):
    
    license_plate_ = ''
    if text=="":
        return ""
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_char_to_int,
               2: dict_char_to_int, 3: dict_char_to_int,7:dict_char_to_int,8:dict_char_to_int,9:dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6,7,8,9]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    text=""
    score=0
    for detection in detections:
        
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score
    
    
    # return text,score
    return None, None


def get_car(license_plate, vehicle_track_ids):
    
    x1, y1, x2, y2, score, class_id = license_plate
    c1=int((x1+x2)/2)
    c2=int((y1+y2)/2)
    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        
        if c1 > xcar1 and c2 > ycar1 and c1 < xcar2 and c2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    return -1, -1, -1, -1, -1
