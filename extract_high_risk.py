import csv

def extract_high_risk(input_csv="detection_results.csv", output_txt="high_risk_paragraphs.txt"):
    high_risk_paragraphs = []
    total = 0
    high_risk_count = 0
    probs = []

    # 读取 CSV 文件
    with open(input_csv, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total += 1
            prob = float(row["AI生成概率"])
            probs.append(prob)
            if row["风险等级"] == "高风险":
                high_risk_count += 1
                # 保留段落编号，方便定位原文
                high_risk_paragraphs.append((int(row["段落编号"]), row["段落内容"], prob))

    # 按段落编号排序，确保和原文顺序一致
    high_risk_paragraphs.sort(key=lambda x: x[0])

    # 计算整体统计
    avg_prob = sum(probs) / len(probs) if probs else 0
    high_risk_ratio = high_risk_count / total if total > 0 else 0

    # 写入 TXT 文件
    with open(output_txt, "w", encoding="utf-8") as f:
        # 写汇总统计
        f.write("📊 汇总统计\n")
        f.write(f"总段落数: {total}\n")
        f.write(f"平均AI生成概率: {avg_prob:.2f}\n")
        f.write(f"高风险段落数量: {high_risk_count}\n")
        f.write(f"高风险段落比例: {high_risk_ratio:.2%}\n")
        f.write("\n============================\n\n")

        # 写目录索引
        f.write("📑 高风险段落目录索引（编号清单）:\n")
        index_list = [str(para[0]) for para in high_risk_paragraphs]
        f.write(", ".join(index_list) + "\n\n")
        f.write("============================\n\n")

        # 写高风险段落内容
        f.write("以下为高风险段落内容（按原文顺序排列）:\n\n")
        for para in high_risk_paragraphs:
            f.write(f"第{para[0]}段: {para[1]} (AI概率={para[2]:.2f})\n\n")

    print(f"✅ 已提取 {high_risk_count} 个高风险段落，并生成目录索引到 {output_txt}")

if __name__ == "__main__":
    extract_high_risk()
