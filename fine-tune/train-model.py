from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

dataset = load_from_disk("vietnews_chunked_data") 
tokenizer = AutoTokenizer.from_pretrained("vietnamese-gpt2-tokenizer")

model = AutoModelForCausalLM.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer)) #resize lại số lượng token của mô hình cho khớp với tokenizer mới 52000

# set cấu hình train 
training_args = TrainingArguments(
    output_dir="./vietnamese-gpt2-model-new", 
    num_train_epochs=4,             #số lần đọc đi đọc lại toàn bộ dữ liệu
    per_device_train_batch_size=4,  #số đoạn văn nhồi vào cùng 1 lúc
    save_steps=1000,                
    save_total_limit=2,             
    logging_steps=100,              # 100 bước in màn hình 1 lần
    prediction_loss_only=True,
    fp16=False,                  #true nếu có card màn hình (GPU)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset, 
)
trainer.train()

trainer.save_model("./vietnamese-gpt2-model-new")
print("done and model saved to 'vietnamese-gpt2-model-new'.")