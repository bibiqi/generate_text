import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("vietnamese-gpt2-tokenizer")
model = AutoModelForCausalLM.from_pretrained("./vietnamese-gpt2-model-new")

def calculate_perplexity(model, tokenizer, text):
    """
    Tính Perplexity (PPL) cho một chuỗi văn bản.
    PPL càng thấp (tiệm cận 1) chứng tỏ mô hình dự đoán càng tốt.
    """
    encodings = tokenizer(text, return_tensors="pt") # mã hóa text thành tensor
    
    input_ids = encodings.input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    #tắt tính gradient để tránh waste bộ nhớ 
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss #CE = -log(P(x)) avg trên all token, càng thấp càng tốt
        
        # Perplexity = e^(Loss)
        perplexity = torch.exp(loss)
        
    return perplexity.item()

def calculate_distinct_n(text, n=1):
    """
    Tính chỉ số Distinct-N để đánh giá độ đa dạng từ vựng và hiện tượng lặp từ.
    - n=1 (Distinct-1): Tỷ lệ các từ đơn duy nhất.
    - n=2 (Distinct-2): Tỷ lệ các cụm 2 từ (bigram) duy nhất.
    Chỉ số càng gần 1.0 (100%) thì văn bản càng đa dạng, không bị nói lắp.
    """
    # tách văn bản thành ds các từ (dựa trên khoảng trắng)
    tokens = text.lower().split()
    total_tokens = len(tokens)
    
    # tránh lỗi chia cho 0 nếu văn bản quá ngắn
    if total_tokens < n:
        return 0.0
    
    # tạo list các n-gram
    ngrams = []
    for i in range(total_tokens - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams.append(ngram)
    
    # count n-gram duy nhất = chuyển thành set (tập hợp)
    unique_ngrams = set(ngrams)
    
    distinct_score = len(unique_ngrams) / total_tokens  #Tổng số n-gram duy nhất / Tổng số token sinh ra
    return distinct_score

# generated_text = "Hệ thống AI này rất thông minh. Hệ thống AI này rất nhanh."
# d1 = calculate_distinct_n(generated_text, n=1)
# d2 = calculate_distinct_n(generated_text, n=2)
# print(f"Distinct-1: {d1:.4f}") # Đo vốn từ vựng
# print(f"Distinct-2: {d2:.4f}") # Đo mức độ lặp cụm từ

def evaluate_model_generation(model, tokenizer, list_of_prompts):
    total_ppl = 0
    total_d1 = 0
    total_d2 = 0
    num_samples = len(list_of_prompts)
    
    for prompt in list_of_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(
            inputs, 
            max_length=100, 
            do_sample=True, 
            top_k=40, 
            temperature=0.7,
            repetition_penalty=1.2
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        total_ppl += calculate_perplexity(model, tokenizer, generated_text)
        total_d1 += calculate_distinct_n(generated_text, n=1)
        total_d2 += calculate_distinct_n(generated_text, n=2)
        
    avg_ppl = total_ppl / num_samples
    avg_d1 = total_d1 / num_samples
    avg_d2 = total_d2 / num_samples
    
    print("=== KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ ===")
    print(f"Perplexity trung bình: {avg_ppl:.2f}")
    print(f"Distinct-1 trung bình: {avg_d1:.4f}")
    print(f"Distinct-2 trung bình: {avg_d2:.4f}")

#Test thử
prompts_test = ["Đội tuyển bóng đá quốc gia", "Thị trường bất động sản", "Công nghệ trí tuệ nhân tạo"]
evaluate_model_generation(model, tokenizer, prompts_test)