from datasets import load_dataset
from transformers import AutoTokenizer

# Train tokenizer TV
dataset = load_dataset("nam194/vietnews", split="train")

batch_size = 1000

def get_training_corpus():
    for i in range(0, len(dataset), batch_size):
        samples = dataset[i : i + batch_size]["article"]
        yield [text for text in samples if isinstance(text, str)]

print("training new tokenizer from GPT-2...")
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = 52000 
new_tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=vocab_size)

new_tokenizer.save_pretrained("vietnamese-gpt2-tokenizer")
print("done!\n")

# trans text to number - TOKENIZE DATASET
print("reload new tokenizer ")
tokenizer = AutoTokenizer.from_pretrained("vietnamese-gpt2-tokenizer")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["article"], truncation=True, max_length=512)

print("convert dataset to input_ids")
tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,          
    num_proc=4,            
    remove_columns=dataset.column_names 
)
print("convert done!\n")

print("after conversion:")
print(tokenized_datasets[0])

# Saving dataset to disk
print("\nSaving tokenized dataset to disk.")
tokenized_datasets.save_to_disk("vietnews_tokenized_data")
print("Done and available in 'vietnews_tokenized_data'.")