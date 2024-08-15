from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

PATH = "dandelin/vilt-b32-finetuned-vqa"
processor = ViltProcessor.from_pretrained(PATH)
model = ViltForQuestionAnswering.from_pretrained(PATH)


def answer_pipline(text: str, image: Image):

    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]
