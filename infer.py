from PIL import Image
from Remote_Sensing_Analysis.ImageProcessor import ImageProcessor


def test_img():
    processor = ImageProcessor(output_folder="results")
    im1 = Image.open("tests/test.jpg")  
    report, percentage = processor.inference(im1)

test_img()