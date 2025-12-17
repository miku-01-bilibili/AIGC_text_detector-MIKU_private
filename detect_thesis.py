from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 选择中文检测器 Zh-v3
MODEL_NAME = "yuchuantian/AIGC_detector_zhv3"

# 加载模型和分词器
print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def detect_text(text):
    """检测单段文字的 AI 生成概率"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  # 第二个类别通常是 AI 生成的概率

def detect_file(file_path):
    """逐段检测 TXT 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        prob = detect_text(line)
        print(f"第{i}段: AI生成概率 = {prob:.2f}")

if __name__ == "__main__":
    # 修改这里为你的论文 TXT 文件名
    FILE_PATH = "my_thesis.txt"
    detect_file(FILE_PATH)
