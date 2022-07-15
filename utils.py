import numpy as np
import math

def add_depth(detections, use_depth=True):
    # for a in data_sequence:
    result = np.empty((len(detections), 6))

    for i in range(len(result)):
        box = detections[i]['bounding_box']
        result[i][0] = box[0]
        result[i][1] = box[1]
        result[i][2] = 0
        result[i][3] = box[2]
        result[i][4] = box[3]
        if(detections[i].get('depth')) and use_depth:
            result[i][5] = detections[i]['depth']
        else:
            result[i][5] = 1

    return result


def get_velocity(box_current, box_previous):

    x_current = (box_current[0]+box_current[3])/2
    y_current = (box_current[1]+box_current[4])/2
    x_prev = (box_previous[0]+box_previous[3])/2
    y_prev = (box_previous[1]+box_previous[4])/2
    return x_current-x_prev, y_current-y_prev

    


def get_velocity_endpoint(n1,n2,v_x1,v_y1):
    v_x1 = v_x1 + 0.00000001 # small value is added so that we dont get infinity
    v_y1 = v_y1 + 0.00000001
    v_length=(math.sqrt(v_x1**2+v_y1**2)*2)*3
    if(v_x1>0 and v_y1<0):
        theta=(math.atan(v_y1/v_x1))
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)
    elif(v_x1<0 and v_y1<0):
        theta=math.atan(v_y1/v_x1)-math.pi
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)
    elif(v_x1<0 and v_y1>0):
        theta=math.pi+(math.atan(v_y1/v_x1))
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)
    else:
        theta=math.atan(v_y1/v_x1)
        return n1+v_length*math.cos(theta) , n2+v_length*math.sin(theta)

def get_box_center(box):
    x= (box[0]+box[3])/2
    y= (box[1]+box[4])/2
    return x,y

def average_box_data(previous,current):
    for i in range(len(previous)):
        current[i]=(current[i]+previous[i])/2
    return current


