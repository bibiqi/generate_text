from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("vietnamese-gpt2-tokenizer")
model = AutoModelForCausalLM.from_pretrained("./vietnamese-gpt2-model-new")

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    device=-1  # -1 chạy bằng CPU
)

@app.route('/')
def home():
    return render_template('newpage.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Nhận dữ liệu từ web gửi lên
    data = request.json
    prompt = data.get('prompt', '')
    
    # Lấy các thông số siêu tham số từ giao diện (có giá trị mặc định an toàn)
    max_length = int(data.get('max_length', 80))
    temperature = float(data.get('temperature', 0.4))
    top_k = int(data.get('top_k', 40))
    repetition_penalty = float(data.get('repetition_penalty', 1.2))

    try:
        outputs = generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True
        )
        
        raw_text = outputs[0]["generated_text"]
        clean_text = raw_text.replace("_", " ")
        
        return jsonify({'success': True, 'text': clean_text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)