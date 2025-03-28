import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, cohen_kappa_score, roc_auc_score
)

# Category 1 files 
category_1_files = {
    "ChatGPT (No Prompt)": "ChatGPT No Prompt 1.csv",
    "ChatGPT (Prompted)": "ChatGPT Prompt 1.csv",
    "LLaMA (No Prompt)": "Llama No Prompt1.csv",
    "LLaMA (Prompted)": "Llama Prompt 1.csv",
    "Gemma (No Prompt)": "Gemma No Prompt 1.csv",
    "Gemma (Prompted)": "Gemma Prompt 1.csv"
}

print(" CATEGORY 1 EVALUATION\n")
results = []  # Collect all results here

# Analyse each file 
for model_name, file in category_1_files.items():
    try:
        df = pd.read_csv(file)

        # Drop image column if it exists
        if 'image' in df.columns:
            df = df.drop(columns=['image'])

        y_true = [1] * len(df)  # All queries expected to be correct
        y_pred = df['label'].apply(lambda x: 1 if float(x) == 1 else 0).tolist()

        # Compute metrics
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = "Not computable"

        print(f" {model_name}")
        print("Confusion Matrix:")
        print(f"  TP: {cm[0][0]}, FN: {cm[0][1]}, FP: {cm[1][0]}, TN: {cm[1][1]}")
        print(f"  Precision: {precision:.2f}")
        print(f"  Recall:    {recall:.2f}")
        print(f"  F1 Score:  {f1:.2f}")
        print(f"  Cohen's Kappa: {kappa:.2f}")
        print(f"  ROC AUC: {auc}")
        print("-" * 40)

        results.append({
            "Model": model_name,
            "TP": cm[0][0],
            "FN": cm[0][1],
            "FP": cm[1][0],
            "TN": cm[1][1],
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Cohen's Kappa": kappa,
            "ROC AUC": auc
        })

    except Exception as e:
        print(f" Error processing {file}: {e}")
        print("-" * 40)

# Save results to CVS 
results_df = pd.DataFrame(results)
results_df.to_csv("Category 1 Analysis Results.csv", index=False)
print("Results saved as CSV: Category 1 Analysis Results.csv")
