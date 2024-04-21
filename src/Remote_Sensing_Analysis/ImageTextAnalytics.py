from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import torch

class ImageTextAnalytics:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.text_model = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # Take the representation of [CLS] token

    def zero_shot_classification(self, description):
        categories = ["rocket launch preparation", "non-preparatory activity"]
        result = self.classifier(description, candidate_labels=categories)
        return result

    def compare_embeddings(self, description, known_phrases):
        embeddings_known = self.sentence_model.encode(known_phrases)
        embedding_caption = self.sentence_model.encode(description)
        similarities = util.pytorch_cos_sim(embedding_caption, embeddings_known)
        return similarities, known_phrases

    def summarize(self, captions):
        text_to_summarize = " ".join(captions)
        text_to_summarize = text_to_summarize[:2000]  # Truncate to prevent overflow
        summary = self.summarizer(text_to_summarize, max_length=100, min_length=15, do_sample=False)
        return summary[0]['summary_text']

    def extract_semantics(self, captions, known_phrases):
        summary = self.summarize(captions)
        zs_classification = self.zero_shot_classification(summary)
        embedding_similarity, known_phrases = self.compare_embeddings(summary, known_phrases)
        report = f"Description:\n {summary}\nZero-shot Classification:\n"
        for i in range(len(zs_classification['labels'])):
            report += f"{zs_classification['labels'][i]}: {100 * float(zs_classification['scores'][i]):.2f}%\n"
        report += "\nEmbedding Similarity to Rocket Prep:\n"
        for i in range(len(known_phrases)):
            report += f"{known_phrases[i]}: {100 * float(embedding_similarity[0][i]):.2f}%\n"
        
        return report, 100 * float(zs_classification['scores'][0])

# Example usage:
# analytics = ImageTextAnalytics()
# captions = ["Rocket on the launch pad.", "Personnel doing final checks."]
# report = analytics.extract_semantics(captions)
# print(report)
