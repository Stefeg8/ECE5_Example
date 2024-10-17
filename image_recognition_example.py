import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO("Yolo files/yolov10n.pt")

class_names_list= None
with open('Yolo files/coco.names', 'r') as f:
    class_names_list = [line.strip() for line in f.readlines()]

# Open video file
input_video_path = "input video path mp4"
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
# I stole this part from chatgpt or some article iirc i kinda forgot where i got this from
output_video_path = "output video path"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform object detection using YOLO
    results = model(frame)[0]
    
    # Convert YOLO results to Supervisely detections format
    detections = sv.Detections.from_ultralytics(results)
    
    # Update the `detections` object to replace class IDs with class names
    num_detections = len(detections.class_id)
    
    # Create a new array for class names with sufficient length
    class_name_array = np.empty(num_detections, dtype='U20')  # Allow up to 20 characters per class name
    
    for i in range(num_detections):
        class_id = detections.class_id[i]
        if class_id < len(class_names_list):
            class_name_array[i] = class_names_list[class_id]
    
    # Update the detections data
    detections.data['class_name'] = class_name_array
    
    # Extract bounding box coordinates and updated class names
    xyxy = detections.xyxy
    class_names = detections.data['class_name']
    
    # Create a list to store centers of detected people in the current frame
    current_frame_people_centers = []

    # Calculate the centers of detected people
    for i in range(len(xyxy)):
        if class_names[i] == "person":
            x_min, y_min, x_max, y_max = xyxy[i]
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            current_frame_people_centers.append((center_x, center_y))
            str_center_x = str(center_x)
            str_center_y = str(center_y)
            print("Detected person at " + str_center_x + " " + str_center_y)


    # Initialize annotators
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    # Annotate bounding boxes and labels on the frame
    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
    
    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
