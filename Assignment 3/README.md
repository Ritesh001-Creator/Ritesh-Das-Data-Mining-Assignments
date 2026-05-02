# Iris Species Classification using K-Nearest Neighbors (KNN)

This notebook demonstrates a basic classification task using the Iris dataset and the K-Nearest Neighbors (KNN) algorithm.

## Steps Performed:

1.  **Data Loading**: The Iris dataset is loaded, containing features like sepal and petal dimensions, and target labels for three species: Setosa, Versicolor, and Virginica.
2.  **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets to evaluate the model's performance on unseen data.
3.  **Model Training**: A K-Nearest Neighbors classifier is initialized with `K=3` (meaning it considers 3 nearest neighbors for classification) and trained using the training data.
4.  **Prediction**: The trained model makes predictions on the test set.
5.  **Model Evaluation**: The model's performance is evaluated using:
    *   **Accuracy Score**: Measures the proportion of correctly classified instances.
    *   **Confusion Matrix**: A table used to describe the performance of a classification model on a set of test data for which the true values are known.

## Results:

*   **Accuracy with K=3**: 100.00%
*   The confusion matrix visually confirms that all test instances were correctly classified, indicating a perfect performance on this particular test split.
## Confusion Matrix
<img width="640" height="547" alt="image" src="https://github.com/user-attachments/assets/ae6b6222-e10a-4c05-9c72-5983ab5557e7" />
