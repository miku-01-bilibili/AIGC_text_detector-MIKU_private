import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 选择中文检测器 Zh-v3
MODEL_NAME = "yuchuantian/AIGC_detector_zhv3"

print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def detect_text(text):
    """检测单段文字的 AI 生成概率"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0][1].item()  

def detect_file(file_path, output_csv="detection_results.csv"):
    """逐段检测 TXT 文件并保存结果，自动标记风险等级"""
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        prob = detect_text(line)
        # 风险等级标记
        if prob >= 0.9:
            risk = "高风险"
        elif prob >= 0.5:
            risk = "中风险"
        else:
            risk = "低风险"

        results.append((i, line, prob, risk))
        print(f"第{i}段: AI生成概率 = {prob:.2f} ({risk})")

    # 计算整体统计
    probs = [r[2] for r in results]
    avg_prob = sum(probs) / len(probs)
    high_risk = sum(1 for p in probs if p >= 0.9)
    high_risk_ratio = high_risk / len(probs)

    print("\n📊 汇总结果：")
    print(f"平均AI生成概率: {avg_prob:.2f}")
    print(f"高风险段落数量: {high_risk} / {len(probs)}")
    print(f"高风险段落比例: {high_risk_ratio:.2%}")

    # 保存到 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["段落编号", "段落内容", "AI生成概率", "风险等级"])
        writer.writerows(results)

    print(f"\n✅ 检测结果已保存到 {output_csv}")

if __name__ == "__main__":
    FILE_PATH = "my_thesis.txt"  # 你的论文 TXT 文件
    detect_file(FILE_PATH)
