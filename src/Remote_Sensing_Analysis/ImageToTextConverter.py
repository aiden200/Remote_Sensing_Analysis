from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModel, AutoTokenizer
from transformers import VisualBertModel, VisualBertConfig
from transformers import LxmertTokenizer, LxmertModel
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class ImageToTextConverter:
    def __init__(self, model_type="CPM-V-2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.model, self.processor = self.load_model_and_processor()

    def load_model_and_processor(self):
        if self.model_type == "blip":
            processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
            model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(self.device)
        elif self.model_type == "CPM-V-2":
            model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device, dtype=torch.bfloat16)
            processor = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
        elif self.model_type == "vis_bert":
            model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre", config=VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")).to(self.device)
            processor = None
        elif self.model_type == "lxmert":
            model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased').to(self.device)
            processor = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        elif self.model_type == "clip":
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            raise ValueError("Unsupported model type")
        return model, processor

    def img_to_text(self, segmented_objects):
        sat_context = "This satellite image shows a rocket launch site. The area may include rockets, launch pads, vehicles, equipment sheds, and fuel tanks, commonly used in preparations for a rocket launch."
        context, images = [], []

        for category, image in segmented_objects:
            combined_context = f"{sat_context} The object appears to be a {category}."
            context.append(combined_context)
            images.append(image)

        captions = []
        if self.model_type == "blip":
            inputs = self.processor(text=context, images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.generate(**inputs, max_length=256, num_beams=3)
            captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
        elif self.model_type == "CPM-V-2":
            self.model.eval()
            question = 'What object does this aerial image contain?'
            def process_image(image):
                msgs = [{'role': 'user', 'content': question}]
                res, context, _ = self.model.chat(
                    image=image,
                    msgs=msgs,
                    context=None,
                    tokenizer=self.processor,
                    sampling=True,
                    temperature=0.7
                )
                return res
            captions = [process_image(image) for image in images]
            
        elif self.model_type == "clip":
            inputs = self.processor(text=context, images=images[0], return_tensors="pt", pad_to_max_length=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            best_caption_index = probs.argmax()
            captions.append(context[best_caption_index])

        # Additional cases as in your original function
        return captions

# # Usage example
# converter = ImageToTextConverter(model_type="blip")
# segmented_objects = [(category, image) for category, image in [("ship", Image.open("path_to_ship_image.jpg"))]]
# captions = converter.img_to_text(segmented_objects)
# print(captions)


