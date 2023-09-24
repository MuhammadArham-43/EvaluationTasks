# Training YOLOv5 on Custom Dataset

<p>Used RoboFlow to convert annotations to YOLO Format</p>
<p>class_id x_center y_center width height</p>

<br>

<p>Used Training Script in Yolo repository to train 200 epochs with YOLOv5s pretrained weights. The results on training and tests set are available in yolov5/runs directory.</p>

## To Regenerate Results

<ol>
    <li>Change Paths for Image Directories in the data/data.yaml file</li>
    <li>Run the train.py file in yolo directory with appropriate command line arguments</li>
    <li>Run detect.py to the test dataset to view results.</li>
</ol>
