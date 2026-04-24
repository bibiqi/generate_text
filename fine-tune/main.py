from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("vietnamese-gpt2-tokenizer")
model = AutoModelForCausalLM.from_pretrained("./vietnamese-gpt2-model-new")

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=-1  # -1 chạy bằng CPU
)

prompt = "Tình hình thời tiết tại Hà Nội được dự đoán sẽ có mưa lớn"

print(f"\n=> Prompt: '{prompt}'")
print("-" * 50)

outputs = generator(
    prompt,
    max_length=50,           #độ dài tối đa ~ số lượng token
    num_return_sequences=3,   #số kết quả muốn sinh ra để chọn lựa
    temperature=0.7,          #độ sáng tạo
    top_k=40,                 #lấy k từ có xsuất đúng cao nhất để ghép
    repetition_penalty=1.5,   #phạt lặp từ
    do_sample=True            #bốc từ random theo xsuất thay vì học vẹt
)

raw_text = outputs[0]["generated_text"]
clean_text = raw_text.replace("_", " ")

print("KẾT QUẢ:")
print(clean_text)