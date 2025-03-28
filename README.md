# Sleep Project – Model Evaluation
This project analyses outputs from various language models (ChatGPT, LLaMA, and Gemma) in relation to sleep-related queries. It evaluates how accurately each model responded using key classification metrics.

## Programming Concepts Demonstrated
This script showcases fundamental programming skills, including:

- Use of loops (`for`)
- Conditional logic (`if`, `try/except`)
- Working with arrays/lists
- Implementation of functions from `sklearn.metrics`
- Basic data cleaning using `pandas`

## Tools & Libraries Used
- Python 3
- pandas
- scikit-learn

## What the Script Does
- Reads prediction results from multiple `.csv` files
- Removes unnecessary columns (e.g., image column)
- Assumes all outputs should be correct (labelled as 1)
- Calculates and prints the following metrics for each model:
  - Confusion matrix
  - Precision, recall, and F1 score
  - Cohen’s Kappa
  - ROC AUC (where computable)
- Outputs a summary CSV file with all results

## Files
- `.csv` files contain model outputs
- Output: `Category 1 Analysis Results.csv`
