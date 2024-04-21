from transformers import AutoTokenizer, AutoModel
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

from PIL import Image
import torch




# Preprocess and encode text
def encode_text(text, tokenizer, text_model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # Take the representation of [CLS] token


def zero_shot_classification(description):
    classifier = pipeline("zero-shot-classification")
    # Define the categories
    categories = ["rocket launch preparation", "non-preparatory activity"]
    result = classifier(description, candidate_labels=categories)
    return result


def compare_embeddings(description):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    known_phrases = [
        "Rocket positioned on the launch pad for final countdown",
        "Final checks on the launch systems",
        "Lots of Activity in the Image",
        "Rocket being fueled"
    ]
    embeddings_known = model.encode(known_phrases)
    embedding_caption = model.encode(description)

    # Calculate similarities
    similarities = util.pytorch_cos_sim(embedding_caption, embeddings_known)
    # print(similarities)
    return similarities, known_phrases


def summarize(captions):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text_to_summarize = " ".join(captions)
    text_to_summarize = text_to_summarize[:300]
    summary = summarizer(text_to_summarize, max_length=100, min_length=15, do_sample=False)
    return summary[0]['summary_text']

def extract_semantics(captions):
    summary = summarize(captions)
    zs_classification = zero_shot_classification(summary)
    embedding_similarity, known_phrases = compare_embeddings(summary)
    report =f"Description:\n {summary}\n"
    report = report + "Zero-shot Classification:\n"
    for i in range(len(zs_classification['labels'])):
        report = report + f"{zs_classification['labels'][i]}: {100 * float(zs_classification['scores'][i]):.2f}%\n"
    
    report = report + "\nEmbedding Similarity to Rocket Prep:\n"
    for i in range(len(known_phrases)):
        report = report + f"{known_phrases[i]}: {100 * float(embedding_similarity[0][i]):.2f}%\n"
    
    return report
