from PIL import Image
from src.Remote_Sensing_Analysis.ImageProcessor import ImageProcessor


def test_img():
    processor = ImageProcessor(output_folder="results")
    im1 = Image.open("tests/test.jpg")  
    processor.inference(im1)

test_img()