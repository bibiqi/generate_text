from datasets import load_dataset
import re
from tqdm import tqdm

#load data
dataset = load_dataset("nam194/vietnews", split="train")
print(dataset[0])

#tien xly
def clean_text(text):
    text = text.strip()
    text = text.replace("_", " ")  
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s.,!?;:()\-\"'%/]", "", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"(\d+)\s*(đ|vnđ|₫)", r"\1 đồng", text, flags=re.IGNORECASE)
    return text.lower()

def clean_text(text):
    text = text.strip()
    text = text.replace("_", " ") # giữ từ ghép
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\d+)\s*(đ|vnđ|₫)", r"\1 đồng", text, flags=re.IGNORECASE) # chuẩn hóa tiền

    # chuẩn hóa số (tuỳ chọn)
    # text = re.sub(r'\d+', '<num>', text)

    text = re.sub(r"[^a-zA-ZÀ-ỹ0-9\s.,!?;:()\-\"'%/]", "", text)
    text = re.sub(r"\.{2,}", ".", text)

    return text.lower()


def sentence_split(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " <eos> ".join(sentences)

def preprocess(example):
    title = clean_text(example.get("title", ""))
    abstract = clean_text(example.get("abstract", "")) 
    article = clean_text(example.get("article", ""))

    if len(title) == 0 and len(abstract) == 0 and len(article) == 0:
        return {"text": ""}

    text = title + ". " + abstract + article
    return {"text": text}

processed_dataset = dataset.map(
    preprocess,
    remove_columns=dataset.column_names
)

# loại dòng rỗng
processed_dataset = processed_dataset.filter(
    lambda x: len(x["text"].strip()) > 0
)

print("preprocessing done")
print(processed_dataset[0])

OUTPUT_FILE = "vi_data.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in tqdm(processed_dataset):
        f.write(item["text"] + "\n")

print(f"file data: {OUTPUT_FILE}")
