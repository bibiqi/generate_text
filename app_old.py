from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.to(device)
model.eval()

# def generate_text(prompt, max_new_tokens=50, temperature=0.6):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_new_tokens=max_new_tokens,
#         temperature=temperature,
#         repetition_penalty=1.2,
#         no_repeat_ngram_size=3,
#         top_k=50,
#         top_p=0.85,
#         do_sample=True,
#         num_beams=5,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_text(prompt, max_new_tokens=50, mode="balanced"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    if mode == "serious":
        temp = 0.3
        k = 40
        p = 0.85
        rep_penalty = 1.15
        nb = 5
    elif mode == "creative":
        temp = 1.2
        k = 100
        p = 0.98
        rep_penalty = 1.3
        nb = 1
    else: # balanced - set default
        temp = 0.7
        k = 50
        p = 0.95
        rep_penalty = 1.2
        nb = 3

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_k=k,
        top_p=p,
        repetition_penalty=rep_penalty,
        no_repeat_ngram_size=3,
        do_sample=True,      
        num_beams=nb,        
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()

    prompt = data.get("prompt", "")
    max_new_tokens = data.get("max_new_tokens", 50)
    mode = data.get("mode", "balanced")
    
    start_time = time.time()

    result = generate_text(
        prompt,
        max_new_tokens=max_new_tokens,
        mode=mode
    )

    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return jsonify({
        "generated_text": result,
        "execution_time": execution_time 
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)