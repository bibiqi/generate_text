from datasets import load_from_disk

tokenized_datasets = load_from_disk("vietnews_tokenized_data")

# Quy định độ dài của mỗi khối (block) - Thường là 128, 256, hoặc 512. Số càng lớn thì máy học câu càng dài, nhưng tốn RAM/VRAM hơn.
block_size = 128

def group_texts(examples):
    # BƯỚC A: Gom tất cả các mảng số lại thành một dải băng khổng lồ
    # Nối đuôi bài báo 1 với bài báo 2, bài 3...
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    
    # Tính tổng chiều dài của dải băng khổng lồ đó
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # BƯỚC B: Cắt bỏ phần dư thừa ở cuối để chia chẵn cho block_size
    # Ví dụ tổng là 1005 số, block_size là 100 -> cắt bỏ 5 số lẻ tẻ ở cuối đi.
    total_length = (total_length // block_size) * block_size
    
    # BƯỚC C: Chặt các khúc dài bằng block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    
    # BƯỚC D: Tạo "đáp án" cho mô hình học
    # Với mô hình GPT, đáp án (labels) chính là bản sao của đầu vào (input_ids)
    result["labels"] = result["input_ids"].copy()
    
    return result


# dùng hàm để bắt đầu chia nhỏ dữ liệu đã được token hóa thành các khối có độ dài block_size
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000, # Xử lý từng mẻ 1000 bài báo một lúc
    num_proc=4,      # Dùng 4 lõi CPU để chạy nhanh
)

print("\nSplit done!")

# check xem các khối đều dài 128 số không
print("Chiều dài của khối đầu tiên:", len(lm_datasets[0]["input_ids"]))
print("Chiều dài của khối thứ hai:", len(lm_datasets[1]["input_ids"]))

lm_datasets.save_to_disk("vietnews_chunked_data")
print("done - available on 'vietnews_chunked_data' folder")