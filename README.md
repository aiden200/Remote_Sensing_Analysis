# Remote Sensing Analysis

This package provides tools for processing and analyzing satellite imagery, utilizing advanced machine learning techniques for object detection, image enhancement, and text analytics from images.

## Installation

To install the Remote Sensing Analysis package, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/aiden200/Remote_Sensing_Analysis.git
   cd Remote_Sensing_Analysis
   ```

2. **Install the Package**:
   Use pip to install the package in editable mode, which is particularly useful for making changes to the code and testing.

   ```bash
   pip install -e .
   ```

3. **Download Model Weights**:
   The package requires specific model weights to function correctly. Download the model weights from the following Google Drive link:
   [Download Model Weights](https://drive.google.com/file/d/1KL3H-Fe1SVoCEFaO4KM4J_FMRF4ocoCz/view?usp=sharing)

   After downloading, place the weights under the `pretrained` folder.

## Usage

### Parameters

When initializing the `ImageProcessor`, you can specify the following parameters:

- **model_weights_path**: Path to the model weights file, default is `"pretrained/YOLOv9_DOTA1_100EPOCHS.pt"`.
- **confidence_threshold**: The confidence threshold for object detection. Objects with a confidence level higher than this threshold are considered. Default is `0.1`.
- **output_folder**: The directory where results will be saved. Default is `"results"`.
- **known_phrases**: A list of phrases against which the descriptions of detected objects will be compared. This helps in identifying specific activities or features in images.

### Example Code

Here is how you can use the `ImageProcessor` in your scripts:

```python
from PIL import Image
from Remote_Sensing_Analysis.ImageProcessor import ImageProcessor

def test_image_processing():
    processor = ImageProcessor(
        model_weights_path="pretrained/YOLOv9_DOTA1_100EPOCHS.pt",
        confidence_threshold=0.1,
        output_folder="results",
        known_phrases=[
            "Rocket positioned on the launch pad for final countdown",
            "Final checks on the launch systems",
            "Lots of Activity in the Image",
            "Rocket being fueled"
        ]
    )
    path = "path_to_your_test_image.jpg"
    im1 = Image.open(path)
    # Using .inference method
    processor.inference(im1)
    # Or using .generate method directly with an image object
    processor.generate(im1)

if __name__ == "__main__":
    test_image_processing()
```

Replace path_to_your_test_image.jpg with the path to the image file you wish to process.
