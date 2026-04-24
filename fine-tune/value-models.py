import torch
import time
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from value import calculate_perplexity, calculate_distinct_n

BASE_MODEL_PATH = "gpt2"
FINETUNED_MODEL_PATH = "./vietnamese-gpt2-model-new" 

def load_prompts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

TEST_PROMPTS = load_prompts_from_file("fine-tune/prompts.txt")

def evaluate_single_setup(model_name_or_path, tokenizer_path, is_sampling, setup_name):
    print(f"\n[{setup_name}] Đang tải...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        total_ppl, total_d1, total_d2 = 0, 0, 0
        total_words = 0
        start_time = time.time()

        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad(): 
                if is_sampling:
                    outputs = model.generate(
                        **inputs,
                        max_length=60,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    outputs = model.generate(
                        **inputs,
                        max_length=60,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=tokenizer.eos_token_id
                    )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Tính toán chỉ số
            total_ppl += calculate_perplexity(model, tokenizer, generated_text)
            total_d1 += calculate_distinct_n(generated_text, n=1)
            total_d2 += calculate_distinct_n(generated_text, n=2)
            total_words += len(generated_text.split())

        num_samples = len(TEST_PROMPTS)
        elapsed_time = time.time() - start_time

        results = {
            "Name": setup_name,
            "PPL": total_ppl / num_samples,
            "D-1": total_d1 / num_samples,
            "D-2": total_d2 / num_samples,
            "Speed": total_words / elapsed_time if elapsed_time > 0 else 0
        }

        # Dọn dẹp bộ nhớ
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return results

    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG] Cấu hình {setup_name} thất bại:")
        print(f"Chi tiết lỗi: {str(e)}") # Dòng này sẽ nói cho bạn biết lỗi gì
        import traceback
        traceback.print_exc() # In bảng tra cứu lỗi đầy đủ
        return None
    
def main():
    print("BẮT ĐẦU CHẠY THỰC NGHIỆM ĐỐI CHỨNG (ABLATION STUDY)")
    print("="*60)
    all_results = []

    res_m1 = evaluate_single_setup("gpt2", "gpt2", False, "M1 (Gốc + Greedy)")
    all_results.append(res_m1)

    res_m2 = evaluate_single_setup("gpt2", "gpt2", True, "M2 (Gốc + Sampling)")
    all_results.append(res_m2)

    res_m3 = evaluate_single_setup(
        "./vietnamese-gpt2-model-new",   # Đường dẫn model
        "./vietnamese-gpt2-tokenizer",   # Đường dẫn tokenizer
        True, 
        "M3 (Fine-tuned + Sampling)"
    )
    all_results.append(res_m3)


    print("\n\n" + "="*80)
    print("BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM".center(80))
    print("="*80)
    print(f"{'Cấu hình Mô hình':<50} | {'PPL (↓)':<10} | {'Dist-1 (↑)':<10} | {'Dist-2 (↑)':<10} | {'Tốc độ (Từ/s)':<15}")
    print("-" * 80)
    
    for r in all_results:
        print(f"{r['Name']:<50} | {r['PPL']:<10.2f} | {r['D-1']:<10.4f} | {r['D-2']:<10.4f} | {r['Speed']:<10.1f}")
    
    print("="*80)

if __name__ == "__main__":
    main()