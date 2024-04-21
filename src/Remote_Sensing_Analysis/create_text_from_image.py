from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import VisualBertModel, VisualBertConfig
from transformers import LxmertTokenizer, LxmertModel
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoModel, AutoTokenizer
from PIL import Image





import torch



def img_to_text(segmented_objects, hf="CPM-V-2"):
    context = []
    images = []
    sat_context = "This satellite image shows a rocket launch site. The area may include rockets, launch pads, vehicles, equipment sheds, and fuel tanks, commonly used in preparations for a rocket launch."

    for category, image in segmented_objects:
        combined_context = f"{sat_context} The object appears to be a {category}."
        # combined_context = "The object in the aerial view is: "
        context.append(combined_context)
        images.append(image)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # hf = "blip"
    if hf == "blip":
        model_id = 'Salesforce/blip-image-captioning-base'
        processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
        inputs = processor(text=context, images=images, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs, max_length=256, num_beams=3)
        captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    elif hf == "CPM-V-2":
        model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True, torch_dtype=torch.bfloat16)
        model = model.to(device='cuda', dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
        model.eval()

        question = 'What object does this aerial image contain?'

        def process_image(image):
            msgs = [{'role': 'user', 'content': question}]
            res, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7
            )
            return res

        captions = [process_image(image) for image in images]
        return captions
    elif hf == "internLM":
        # init model and tokenizer
        model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-4khd-7b', trust_remote_code=True)

        query1 = '<ImageHere>Illustrate the fine details present in the image'
        image = images[0]
        question = '<ImageHere>What object does this aerial image contain?'

        with torch.cuda.amp.autocast():
            response, his = model.chat(tokenizer, query=question, image=image, hd_num=55, history=[], do_sample=False, num_beams=3)
        print(response)


    elif hf == "vis_bert":
        config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre", config=config).to(device)
        outputs = model(**inputs)
    elif hf == "lxmert":
        tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')
        model = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
        inputs = tokenizer("What is in the picture?", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=256, num_beams=3)
        captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    elif hf == 'clip':
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(text=context, images=image, return_tensors="pt", pad_to_max_length=True)
        outputs = model(**inputs)
        # Post-process to find the best description match
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we use softmax to convert logits to probabilities
        best_caption = context[probs.argmax()]

        print("Best caption:", best_caption)





    return captions
