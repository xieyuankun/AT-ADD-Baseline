import csv
import os
# input_csv = "atadd_baseline/ckpt_t2/baseline_ft-w2v2aasist/result/atadd-track2_logits.csv"
base_path = os.path.dirname(input_csv)
output_csv = os.path.join(base_path, "predict.csv")
threshold = 0.5

with open(input_csv, "r", encoding="utf-8-sig", newline="") as fin, \
     open(output_csv, "w", encoding="utf-8", newline="") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)

    writer.writerow(["name", "predict"])

    for row in reader:
        name = row["name"].strip()
        score = float(row["score"])
        predict = "real" if score >= threshold else "fake"
        writer.writerow([name, predict])

print(f"Done. Saved to {output_csv}")
