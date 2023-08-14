Machine Learning 
A Diabetes Prediction program. Dataset sourced from, National Institute of Diabetes and Digestive and Kidney Diseases,program utilizes a decision tree model to generate its predictions.In other word, this program uses real world dataset.
User can Input  health metrics and get an immediate prediction from our model.
summary: The program takes input data from patients, processes it using a predictive model (presumably trained on past data), and then provides an output in terms of these metrics. This aligns with the methodology employed in machine learning applications.
Program prints out to texbox or console( activate from program removing # header)
Model shows : 
Accuracy:
This value informs you about the proportion of predictions made by the model that were correct. A higher value signifies that the model was largely accurate in its predictions.
Interpretation: This helps you understand that accuracy is the ratio of correctly predicted instances to the total instances. A higher value indicates fewer mistakes made by the classifier.

Precision tells you how many of the positive predictions made by the model (i.e., predicting that a person has diabetes) were actually correct.
Interpretation: Precision is the ratio of correctly predicted positive instances to the total predicted positives. A higher value means that when the model predicted diabetes, it was usually right.

Recall measures how many of the actual positive cases (i.e., people who truly have diabetes) were correctly identified by the model.
Interpretation: This metric gauges the proportion of actual positive cases that the model successfully identified. A higher recall means that the model captured most of the relevant cases.
F1 Score:
The F1 Score is a metric that provides an overall assessment of the model's performance by considering both precision and recall. It's particularly useful when the cost of false positives and false negatives are significantly different.
Interpretation: F1 Score is the harmonic mean of precision and recall, offering a balance between them. An F1 Score closer to 1 suggests superior performance.
used libraries and reason: 
Pandas: To load the dataset from a CSV file, preprocess the data, and manipulate data structures
train_test_split: To split the dataset into training and testing sets.
DecisionTreeClassifier: To build the decision tree model.
plot_tree: To visualize the decision tree.
confusion_matrix, ConfusionMatrixDisplay: For creating and visualizing the confusion matrix.
Evaluation metrics (accuracy_score, precision_score, recall_score, f1_score): To evaluate the performance of the decision tree model.

Please feel free to ask. 


