from PIL import Image
from ultralytics import YOLO
import os
import math
import torch

class YOLOInference:
    def __init__(self, model_path, output_folder, confidence_threshold=0.35):
        self.model = YOLO(model_path)
        self.output_folder = output_folder
        self.confidence_threshold = confidence_threshold
        self.mapping = {0: "plane", 1: "ship", 2: "large vehicle", 3: "small vehicle", 4: "helicopter"}

    def run_inference(self, image_path):
        with torch.no_grad():
            results = self.model([image_path])  # Expecting a list of results for each image

        cropped_dir = os.path.join(self.output_folder, "cropped")
        os.makedirs(cropped_dir, exist_ok=True)
        
        sub_images = []
        img = Image.open(image_path)
        width, height = img.size

        result_file = os.path.join(self.output_folder, 'result.jpg')
        for result in results:
            for index, box in enumerate(result.boxes.xyxy):
                if result.boxes.conf[index] < self.confidence_threshold:
                    continue

                category = self.mapping[int(result.boxes.cls[index])]
                x1, y1, x2, y2 = map(int, box)
                
                # Calculate the center and initial crop size
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                crop_size = max(128, x2 - x1, y2 - y1)
                crop_size = max(128, math.ceil(crop_size / 32) * 32)  # Adjust to nearest multiple of 32

                new_x1 = max(0, center_x - crop_size / 2)
                new_y1 = max(0, center_y - crop_size / 2)
                new_x2 = min(width, center_x + crop_size / 2)
                new_y2 = min(height, center_y + crop_size / 2)

                cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
                if cropped_img.mode == 'RGBA':
                    cropped_img = cropped_img.convert('RGB')
                sub_images.append((category, cropped_img))
                output_path = os.path.join(cropped_dir, f"cropped_image_{index}.png")
                cropped_img.save(output_path)

            result.save(filename=result_file, labels=False)

        return sub_images, result_file

