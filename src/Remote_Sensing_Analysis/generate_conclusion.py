from generate_bounding_boxes import load_model_and_run_inference
from create_text_from_image import img_to_text
from compare_semantics import extract_semantics
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
from tqdm import tqdm


import os

def generate_report(orig_img, grid_img, report, output_folder):
    font_path = "assets/DejaVuSans-Bold.ttf"  # Adjust the path to your font
    main_image = Image.open(orig_img)
    grid_image = Image.open(grid_img)

    # Resize main image to make it smaller
    scale_factor = 0.75  # Reduce size by 25%
    main_image = main_image.resize((int(main_image.width * scale_factor), int(main_image.height * scale_factor)))

    # Resize grid image to fit below the main image if necessary
    grid_image = grid_image.resize((main_image.width, int(main_image.width / grid_image.width * grid_image.height)))

    # Create a new blank image with enough space to hold the main image, grid image, and text
    additional_text_space = 300  # Increase space for text
    final_height = main_image.height + grid_image.height + additional_text_space
    final_image = Image.new('RGB', (main_image.width, final_height), 'white')

    # Paste the main image and the grid image onto the final image
    final_image.paste(main_image, (0, 0))
    final_image.paste(grid_image, (0, main_image.height))

    # Draw the text report
    draw = ImageDraw.Draw(final_image)
    font_size = 24  # Larger font size
    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    text_position = (10, main_image.height + grid_image.height + 10)
    draw.text(text_position, "Here is the text report summarizing the observations:\n" + report, fill='black', font=font)

    # Save the final composite image
    sv_path = os.path.join(output_folder, 'final_report.jpg')
    final_image.save(sv_path)
    print(f"Report saved at: {sv_path}")


def extract_central_patch(img_path, patch_size, temp_path):
    # Load image
    img = Image.open(img_path)
    img_array = np.array(img)

    # Calculate the center of the image
    center_y = img_array.shape[0] // 2 + 700
    center_x = img_array.shape[1] // 2

    # Calculate the starting indices of the patch
    start_y = center_y - patch_size[0] // 2
    start_x = center_x - patch_size[1] // 2

    # Extract the central patch
    central_patch = img_array[start_y:start_y + patch_size[0], start_x:start_x + patch_size[1]]

    # Convert the array back to an image
    patch_img = Image.fromarray(central_patch)

    # Save or display the image
    out = os.path.join(temp_path, 'central_patch.jpg')
    patch_img.save(out)
    return out

def generate(image_path, output_folder, confidence_threshold=0.1):
    model_weights_path = "pretrained/best.pt"
    # image_path = 'data/maxar/10_Mar_Blacksky_Xichan_R1C1.jpg'
    # image_path = 'model/datasets/DOTAv1/images/train/P0178.jpg'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # img = Image.open(image_path)
    # f = np.array(img)


    # output_folder = 'temp'

    # image_path = extract_central_patch(image_path, (1000, 1000), output_folder)
    detected_sub_images, result_file = load_model_and_run_inference(model_weights_path, image_path, output_folder, confidence_threshold)
    # (Category, Image object)

    # print(len(detected_sub_images))
    if len(detected_sub_images) == 0:
        return "Nothing Detected"
    
    print(f"{len(detected_sub_images)} objects detected")

    # Enhance each image

    # Text to image given context
    captions = img_to_text(detected_sub_images, hf="CPM-V-2")


    wrapper = textwrap.TextWrapper(width=20)  # Adjust width to suit your layout
    wrapped_captions = [wrapper.fill(text=caption) for caption in captions]
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6))  # Adjust the figsize to fit your screen or requirements
    axes = axes.ravel()

    # Apply text wrapping to captions
    wrapper = textwrap.TextWrapper(width=25)  # Adjust width to suit your layout
    wrapped_captions = [wrapper.fill(text=caption) for caption in captions]

    for i in range(min(10, len(detected_sub_images))):  # Ensure not to exceed 10 images
        ax = axes[i]
        ax.imshow(detected_sub_images[i][1])  # Show the image on the subplot
        ax.set_title(wrapped_captions[i], fontsize=10)  # Use wrapped captions with a suitable font size
        ax.axis('off')  # Turn off axis to not show ticks and labels

    plt.tight_layout()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    grid_img = os.path.join(output_folder, 'my_plot.png')
    plt.savefig(grid_img, dpi=300)  # Save as PNG with high resolution
    # plt.show()

    report = extract_semantics(captions)
    with open(os.path.join(output_folder, "report.txt"), "w") as f:
        f.write(report)
    # print(report)
    generate_report(result_file, grid_img, report, output_folder)

    # Compare in embedding space

    # Return Analysis

def process_images_in_folder(base_path):

    for dirpath, dirnames, filenames in os.walk(base_path):
        for filename in tqdm(filenames):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(dirpath, filename)
                if not os.path.exists("report_folder"):
                    os.mkdir("report_folder")
                
                output_folder = os.path.join("report_folder", filename[:-4])

                generate(image_path, output_folder, confidence_threshold=0.1)



process_images_in_folder("data/maxar_test_data")