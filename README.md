# Gauge Reader
The gauge reader is a tool that enables you to extract the value of a gauge from a picture. It is based on OpenCV and Python.


# How to use it
* Install python3
* Install pip3 install python3-opencv


# How to run it
Change the variables crop_top_left and crop_bottom_right, so that only the gauge is visible in the picture.
You might also need to adjust max_distance_line_from_center, minimum_line_length and max_distance_from_center to get good results
* Run `python3 main.py`

# Credits
This codebase is based on Intel tutorial: https://www.intel.com/content/www/us/en/developer/articles/technical/analog-gauge-reader-using-opencv.html