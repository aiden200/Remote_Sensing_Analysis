from PIL import Image
from ultralytics import YOLO
import os
import math
import torch

def load_model_and_run_inference(model_path, image_path, output_folder, confidence_threshold=0.35):
    # Load the model
    model = YOLO(model_path)

    mapping = {0: "plane", 1:"ship", 2: "large vehicle", 3: "small vehicle", 4: "helicopter"}


    with torch.no_grad():
        results = model([image_path])  # Expecting a list of results for each image
    
    cropped_dir = os.path.join(output_folder, "cropped")
    if not os.path.exists(cropped_dir):
        os.mkdir(cropped_dir)
    
    # Initialize a list to hold cropped sub-images
    sub_images = []
    
    # Load the image once
    img = Image.open(image_path)
    width, height = img.size
    
    # Process each result object (assuming one image, one result object)
    # print(results[0])
    result_file = os.path.join(output_folder, 'result.jpg')
    for result in results:
        # Iterate over bounding boxes

        # print(boxes)
        # print(masks)
        # print(keypoints)
        # print(probs)
        # print(results[0].boxes.conf)
        # print(results[0].boxes.cls)
        # for det in results[0].boxes.xyxy:  # Iterate over detections
        #     x1, y1, x2, y2, conf, cls_id = det
        #     print(f'Class: {results.names[int(cls_id)]}, Confidence: {conf:.2f}')

        for index, box in enumerate(result.boxes.xyxy):
            # Extract coordinates

            if result.boxes.conf[index] < confidence_threshold:
                # print(f"Confidence too low: {result.boxes.conf[index]}")
                continue

            category = mapping[int(result.boxes.cls[index])]

            x1, y1, x2, y2 = map(int, box)
            
            
            # Calculate the center and initial crop size
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            crop_size = max(128, x2 - x1, y2 - y1)
            half_crop_size = crop_size / 2
            
            # Adjust crop size to the nearest multiple of 32
            crop_size = max(128, math.ceil(crop_size / 32) * 32)
            
            # Calculate new bounding box coordinates ensuring they are within the image dimensions
            new_x1 = max(0, center_x - crop_size / 2)
            new_y1 = max(0, center_y - crop_size / 2)
            new_x2 = min(width, center_x + crop_size / 2)
            new_y2 = min(height, center_y + crop_size / 2)
            
            # Crop the image
            cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
            if cropped_img.mode == 'RGBA':
                cropped_img = cropped_img.convert('RGB')
            sub_images.append((category, cropped_img))

            # Save cropped image to the specified output folder
            output_path = os.path.join(cropped_dir, f"cropped_image_{index}.png")
            cropped_img.save(output_path)
        
        result.save(filename=result_file, labels=False) 
    return sub_images, result_file

# # Example usage
# model_weights_path = "model/runs/detect/train8/weights/best.pt"
# image_path = 'model/datasets/DOTAv1/images/train/P0178.jpg'
# output_folder = 'temp'  # Specify your output folder path here
# detected_sub_images = load_model_and_run_inference(model_weights_path, image_path, output_folder)


