# Iris Species Classification

This project classifies flowers in the Iris dataset into three species using two models: K-Nearest Neighbors and Decision Tree.

## Dataset
- **Source**: iris.csv
- Contains:
  - Sepal length & width
  - Petal length & width
  - Species (Target)

## Methods Used
- Standard Scaling and Imputation using Pipelines
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Cross-validation and classification report for evaluation
- Tree visualization for interpretability

## Results
- **Accuracy**: ~100% for both models
- **Cross-validation scores**: Consistent and high
- The dataset is simple and very separable, leading to perfect scores

## Key Learnings
- First experience using Pipelines and ColumnTransformers
- Learned how decision trees visualize decision-making
- Solidified understanding of classification algorithms

---

**Note**: This dataset is very clean and small — great for practice but not realistic. Adding noise or using larger datasets would be the next challenge.
