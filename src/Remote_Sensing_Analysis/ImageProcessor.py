from src.Remote_Sensing_Analysis.YOLOInference import YOLOInference
from src.Remote_Sensing_Analysis.ImageToTextConverter import ImageToTextConverter
from src.Remote_Sensing_Analysis.ImageTextAnalytics import ImageTextAnalytics
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
from tqdm import tqdm


import os


class ImageProcessor:
    def __init__(self, model_weights_path="pretrained/YOLOv9_DOTA1_100EPOCHS.pt", confidence_threshold=0.1, output_folder="results"):
        self.ODM = YOLOInference(model_weights_path, output_folder, confidence_threshold=confidence_threshold)
        self.itt = ImageToTextConverter(model_type="CPM-V-2")
        self.ita = ImageTextAnalytics()
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        

    def generate_report(self, orig_img, grid_img, report):
        font_path = "assets/DejaVuSans-Bold.ttf"
        main_image = Image.open(orig_img)
        grid_image = Image.open(grid_img)

        scale_factor = 0.75
        main_image = main_image.resize((int(main_image.width * scale_factor), int(main_image.height * scale_factor)))
        grid_image = grid_image.resize((main_image.width, int(main_image.width / grid_image.width * grid_image.height)))
        additional_text_space = 300

        final_height = main_image.height + grid_image.height + additional_text_space
        final_image = Image.new('RGB', (main_image.width, final_height), 'white')
        final_image.paste(main_image, (0, 0))
        final_image.paste(grid_image, (0, main_image.height))
        draw = ImageDraw.Draw(final_image)

        font_size = 24
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        text_position = (10, main_image.height + grid_image.height + 10)
        draw.text(text_position, "Here is the text report summarizing the observations:\n" + report, fill='black', font=font)

        sv_path = os.path.join(self.output_folder, 'final_report.jpg')
        final_image.save(sv_path)
        print(f"Report saved at: {sv_path}")

        return sv_path

    def extract_central_patch(self, img_path, patch_size, temp_path):
        img = Image.open(img_path)
        img_array = np.array(img)

        center_y = img_array.shape[0] // 2 + 700
        center_x = img_array.shape[1] // 2
        start_y = center_y - patch_size[0] // 2
        start_x = center_x - patch_size[1] // 2

        central_patch = img_array[start_y:start_y + patch_size[0], start_x:start_x + patch_size[1]]
        patch_img = Image.fromarray(central_patch)
        out = os.path.join(temp_path, 'central_patch.jpg')
        patch_img.save(out)

        return out
    
    def inference(self, img):
        if not os.path.exists("temp"):
            os.mkdir("temp")

        img.save("temp/inf.png") 
        self.generate("temp/inf.png")


    def generate(self, image_path): 

        detected_sub_images, result_file = self.ODM.run_inference(image_path)
        
        if len(detected_sub_images) == 0:
            return "Nothing Detected", 0
        print(f"{len(detected_sub_images)} objects detected")

        captions = self.itt.img_to_text(detected_sub_images)
        wrapper = textwrap.TextWrapper(width=20)
        wrapped_captions = [wrapper.fill(text=caption) for caption in captions]

        rows, cols = 2, 5
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
        axes = axes.ravel()
        for i in range(min(10, len(detected_sub_images))):
            ax = axes[i]
            ax.imshow(detected_sub_images[i][1])
            ax.set_title(wrapped_captions[i], fontsize=10)
            ax.axis('off')

        plt.tight_layout()
        grid_img = os.path.join(self.output_folder, 'my_plot.png')
        plt.savefig(grid_img, dpi=300)

        known_phrases = [
            "Rocket positioned on the launch pad for final countdown",
            "Final checks on the launch systems",
            "Lots of Activity in the Image",
            "Rocket being fueled"
        ]

        report, percentage = self.ita.extract_semantics(captions, known_phrases)

        with open(os.path.join(self.output_folder, "report.txt"), "w") as f:
            f.write(report)

        self.generate_report(result_file, grid_img, report)
        return report, percentage

def process_images_in_folder(base_path):
    processor = ImageProcessor()

    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in tqdm(filenames):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(dirpath, filename)
                if not os.path.exists("report_folder"):
                    os.mkdir("report_folder")
                
                output_folder = os.path.join("report_folder", filename[:-4])
                processor.output_folder = output_folder
                processor.generate(image_path, output_folder, confidence_threshold=0.1)



# process_images_in_folder("data/maxar_test_data")

