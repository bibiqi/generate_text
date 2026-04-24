Fine-tuning GPT-2 trên VietNews Dataset

Dự án fine-tune GPT-2 trên dữ liệu tin tức tiếng Việt (VietNews: https://huggingface.co/datasets/nam194/vietnews)

1. Cài đặt môi trường
pip install -r requirements.txt

2. Huấn luyện Tokenizer
python fine-tune/tokenizer.py

Kết quả:
vietnamese-gpt2-tokenizer/ → tokenizer tiếng Việt
vietnews_tokenized_data/ → dữ liệu đã tokenize 

3. Chia nhỏ dữ liệu (Chunking)
python fine-tune/chunk_data.py

📌 Kết quả:

vietnews_chunked_data/

4. Fine-tune mô hình GPT-2
python fine-tune/train-model.py
 Kết quả:

vietnamese-gpt2-model-new/

 5. Đánh giá mô hình
python fine-tune/value-models.py

 6. Chạy demo
 GPT-2 gốc
python app_old.py
 GPT-2 fine-tuned
python app_new.py
