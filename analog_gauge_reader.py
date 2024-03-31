'''  
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import numpy as np
#import paho.mqtt.client as mqtt
import time
from pathlib import Path 
import pandas as pd

def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

def dist_2_pts(x1, y1, x2, y2):
    #print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def distance_line_from_pt(line, pt):
    # Line is a list of two points, [[x1, y1], [x2, y2]]
    # pt is a list of two points, [x, y]
    # Returns the distance between the line and the point
    x1, y1 = line[0]
    x2, y2 = line[1]
    x, y = pt
    return np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

def computer_line_circle_intersection(line, circle_center, r):
    '''
    Finds the intersection points between a line and a circle
    :param line: A list of two points, [[x1, y1], [x2, y2]]
    :param circle_center: A list of two points, [x, y]
    :param r: The radius of the circle
    :return: A list of two points, [[x1, y1], [x2, y2]]
    '''

    # First convert the line in y= mx + c form
    x1, y1 = line[0]
    x2, y2 = line[1]
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    p, q = circle_center    

    # Now find the intersection points
    # (x - p)^2 + (y - q)^2 = r^2
    # (m * x + c - p)^2 + (x - p)^2 = r^2
    # (m^2 + 1) * x^2 + (2 * m * c - 2 * m * q - 2 * p) * x + (c^2 - 2 * c * q + q^2 - r^2 + p^2) = 0
    # a = m^2 + 1
    # b = (2 * m * c - 2 * m * q - 2 * p)
    # c = (c^2 - 2 * c * q + q^2 - r^2 + p^2)
    
    A = m**2 + 1
    B = 2 * m * c - 2 * m * q - 2 * p
    C = c**2 - 2 * c * q + q**2 - r**2 + p**2
        # Find the discriminant
    discriminant = B**2 - 4 * A * C
    if discriminant < 0:
        return []

    # Find the two roots
    x_interesection_1 = (-B + np.sqrt(discriminant)) / (2 * A)
    x_interesection_2 = (-B - np.sqrt(discriminant)) / (2 * A)
    y_interesection_1 = m * x_interesection_1 + c
    y_interesection_2 = m * x_interesection_2 + c

    return [[x_interesection_1, y_interesection_1], [x_interesection_2, y_interesection_2]]


    

def calibrate_gauge(img, debug=False):
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''

    
    
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gray
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Downsample image for faster processing

    if debug:
        cv2.imwrite("debug/reference_gray.jpg", gray)
    # gray = cv2.medianBlur(gray, 5)

    #for testing, output gray image
    #cv2.imwrite('gauge-%s-bw.%s' %(gauge_number, file_type),gray)

    #detect circles
    #restricting the search from 35-48% of the possible radii gives fairly good results across different samples.  Remember that
    #these are pixel values which correspond to the possible radii search range.
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    x,y,r = avg_circles(circles, b)


    #draw center and circle
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle

    if debug:
        cv2.imwrite("debug/reference_image.jpg",img)

    
    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 10.0 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(img, '%s' %(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)

    if debug:
        cv2.imwrite("debug/reference_image_with_lines.jpg", img)

    #get user input on min, max, values, and units
    min_angle = 52
    max_angle = 312
    min_value = 0
    max_value = 6
    units = "bar"
    
    circle_center = (x, y)
    return min_angle, max_angle, min_value, max_value, units, circle_center, r

def get_current_value(img, min_angle, max_angle, min_value, max_value, circle_center, r, gauge_number, debug=False):   
    # Use the first channel as gray
    gray2 = img[:,:,2]
    #gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The clock hand is black, so we need to invert the image to get a white hand
    gray2 = cv2.bitwise_not(gray2)

    # Threshold the image to get only white pixels
    ret, dst2 = cv2.threshold(gray2, 180, 255, cv2.THRESH_BINARY)

    

    # for testing, show image after thresholding
    if debug:
        cv2.imwrite('debug/gauge-%s-tempdst2.%s' % (gauge_number, "jpg"), dst2)

    # find lines
    minLineLength = 20
    maxLineGap = 0
    lines = cv2.HoughLinesP(image=dst2, rho=1, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=maxLineGap)  # rho is set to 3 to detect more lines, easier to get more then filter them out later

    max_distance_line_from_center = 50
    minimum_line_length = 160#140
    max_distance_from_center = 200
    new_lines = []
    #for testing purposes, show all found lines
    candidate = None
    for i in range(0, len(lines)):
      for x1, y1, x2, y2 in lines[i]:
        line = [[x1, y1], [x2, y2]]
        line_distace_from_center_tolerance = distance_line_from_pt(line, circle_center)
        line_length = dist_2_pts(x1, y1, x2, y2)
        tmp1 = dist_2_pts(x1, y1, circle_center[0], circle_center[1])
        tmp2 = dist_2_pts(x2, y2, circle_center[0], circle_center[1])

        
        
        distance_from_center = min(tmp1, tmp2)
        if line_length > minimum_line_length and distance_from_center < max_distance_from_center and line_distace_from_center_tolerance < max_distance_line_from_center:
            if candidate is None:
                candidate = line
            else:
                # If the line is closer to the center, it is a better candidate
                if distance_line_from_pt(line, circle_center) < distance_line_from_pt(candidate, circle_center):
                    # Show current line
                    # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # cv2.imwrite("test_curr.jpg", img)
                    candidate = line
    
    # Find the point that is more distant from the center
    if not candidate:
        for line in lines:
            x1, y1, x2, y2 = line.squeeze()
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite("debug/test_fin3.jpg", img)
        return -1
    x1, y1 = candidate[0]
    x2, y2 = candidate[1]
    tmp1 = dist_2_pts(x1, y1, circle_center[0], circle_center[1])
    tmp2 = dist_2_pts(x2, y2, circle_center[0], circle_center[1])

    if tmp1 > tmp2:
        furthest_point = [x1, y1]
    else:
        furthest_point = [x2, y2]

    
    # Show the candidate line
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    

    # Find intersection line with circle
    interesection = computer_line_circle_intersection(candidate, circle_center, r)
    # If there are two intersection points, take the one that that is closer to the furthest point
    if len(interesection) == 2:
        first_point = interesection[0]
        second_point = interesection[1]
        # Find the distance from the furthest point
        tmp1 = dist_2_pts(first_point[0], first_point[1], furthest_point[0], furthest_point[1])
        tmp2 = dist_2_pts(second_point[0], second_point[1], furthest_point[0], furthest_point[1])
        if tmp1 > tmp2:
            interesection = [second_point]
        else:
            interesection = [first_point]

    # Show the intersection points
    cv2.circle(img, circle_center, r, (0, 0, 255), 3, cv2.LINE_AA)  # draw circle
    cv2.circle(img, (int(interesection[0][0]), int(interesection[0][1])), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle
    if debug:
        cv2.imwrite("debug/test_fin2.jpg", img)
    
    # Find the angle of the intersection point
    x_angle = interesection[0][0] - circle_center[0]
    y_angle = interesection[0][1] - circle_center[1]

    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    res = np.rad2deg(res)
    res = res%360

    # Find the highest point of the circle
    highest_point = [circle_center[0], circle_center[1] - r]
    # Find the angle of the highest point
    x_angle = highest_point[0] - circle_center[0]
    y_angle = highest_point[1] - circle_center[1]
    # Draw the highest point on the image
    cv2.circle(img, (int(highest_point[0]), int(highest_point[1])), 2, (0, 255, 0), 3, cv2.LINE_AA)  # draw center of circle
    if debug:
        cv2.imwrite("debug/test_fin3.jpg", img)

    # This is our 0 degree point
    x_angle = highest_point[0] - circle_center[0]
    y_angle = highest_point[1] - circle_center[1]
    res2 = np.arctan(np.divide(float(y_angle), float(x_angle)))
    reference = np.rad2deg(res2)%360
    final_angle = res - reference + 180


    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min
    
    return new_value


#

def main():
    crop_top_left = (800,1550)
    crop_bottom_right = (2100,2800)
    reference_image_path = "data/Manometer/reference.jpg"
    reference_image = cv2.imread(reference_image_path)

    measurement_folder = Path("data/Manometer/")
    if crop_top_left is not None and crop_bottom_right is not None:
        reference_image = reference_image[crop_top_left[0]:crop_bottom_right[0],crop_top_left[1]:crop_bottom_right[1]]
        reference_image = cv2.resize(reference_image, (0, 0), fx=0.5, fy=0.5)
    

    
    # name the calibration image of your gauge 'gauge-#.jpg', for example 'gauge-5.jpg'.  It's written this way so you can easily try multiple images
    min_angle, max_angle, min_value, max_value, units, circle_center, r = calibrate_gauge(reference_image, debug=True)

    result = {"path":[],"reading":[],"datetime":[]}
    #feed an image (or frame) to get the current value, based on the calibration, by default uses same image as calibration
    for img_path in measurement_folder.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if crop_top_left is not None and crop_bottom_right is not None:
            img = img[crop_top_left[0]:crop_bottom_right[0],crop_top_left[1]:crop_bottom_right[1]]
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #convert to gra

        val = get_current_value(img, min_angle, max_angle, min_value, max_value, circle_center, r, 1, debug=True)
        print("Current reading: %s %s" %(val, units))
        
        
        try:
            date, hour = img_path.stem.split("_")[1:]
        except:
            continue

        result["path"].append(img_path.name)
        result["datetime"].append(date + " " + hour)
        result["reading"].append(val)

        # # Randomly breakpoint 
        # if np.random.rand() < 0.1:
        #     breakpoint()
    # Sort by datetime
    result = pd.DataFrame(result)
    result = result.sort_values("datetime")
    # convert to yyyy-mm-dd hh:mm:ss
    result["datetime"] = pd.to_datetime(result["datetime"], format="%Y%m%d %H%M%S")
    result = result.iloc[1:]
    result["reading"] = result["reading"].round(2)
    # Set the index to datetime
    result = result.set_index("datetime")
    result.to_csv("readings.csv")
    result.to_excel("readings.xlsx")

    # Create a plot of the readings x the time y the readings with plotly
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["datetime"], y=result["reading"]))
    fig.update_layout(title="Manometer readings", xaxis_title="Datetime", yaxis_title="Reading")
    fig.write_html("readings.html")
    breakpoint()

if __name__=='__main__':
    main()
   	
