Dự án này thực hiện việc Fine-tune mô hình GPT-2 trên bộ dữ liệu tin tức Tiếng Việt (vietnews - https://huggingface.co/datasets/nam194/vietnews)

Thứ tự thực hiện:
- Cài đặt môi trường:
    pip install -r requirements.txt
- Huấn luyện Tokenizer (fine-tune/tokenizer.py): sau khi chạy xong có được thư mục "vietnamese-gpt2-tokenizer" (dịch chữ <-> số)
và "vietnews_tokenized_data" (chuyển chữ thành số để chuẩn bị đem chia nhỏ trước train)
    python tokenizer.py
- Chia nhỏ dữ liệu theo blocks (fine-tune/chunk_data.py): thu được thư mục vietnews_chunked_data
    python chunk_data.py
- Huấn luyện mô hình (fine-tune/train-model.py): load mô hình GPT-2 gốc và huấn luyện lại (Fine-tune) trên dữ liệu tiếng Việt đã chuẩn bị.
Mô hình sau khi học được lưu tại thư mục "vietnamese-gpt2-model-new"
    python train-model.py
- Đánh giá mô hình (fine-tune/value-models.py): so sánh mô hình gốc và mô hình đã fine-tune qua các chỉ số PPL và Distinct-N
    python value-models.py
- Triển khai giao diện:
  + Sử dụng mô hình gốc + sampling: python app_old.py
  + Sử dụng mô hình fine-tune + sampling: python app_new.py
