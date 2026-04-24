from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

text = "Trong bài thi TOEIC, kỹ năng Reading thường là “điểm nghẽn” khiến"
#text ="One way to look at how energy moves from animal to animal is "
ips = tokenizer(text, return_tensors="pt")

ops = model.generate(
    ips["input_ids"],
    #max_length=50, #length = ips + token sau khi sinh 
    max_new_tokens=50, #length = only token sau khi sinh
    do_sample=True, #lay sample random 
    temperature=0.7, #dchinh do nhiem nhieu
    top_k=100 ,#lay 100 token xs cao nhat
    top_p=0.9 ,#lay token co xac suat 0.9
)

print(tokenizer.decode(ops[0], skip_special_tokens=True))
