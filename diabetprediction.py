import pandas as pd
#sklearn modules import
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#for graphics and gui 
import tkinter as tk
from tkinter import ttk,Canvas,PhotoImage
from PIL import Image, ImageTk
#for graphic
import matplotlib.pyplot as plt
#pandas for data read
import pandas as pd
import io 

# open csv document
df = pd.read_csv('diabetes.csv')

# data and column correction if there is any faulty column
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# extracting the first 8 columns as features
feature_column_names = df.columns[:8].tolist()
X = df[feature_column_names]

# using the 9th column as the output
output_column_name = df.columns[8]
y = df[output_column_name]

# data inspection for test can be deactivated later
print('First few rows of the dataset for test')
print(df.head())

print("\nFeature columns:", feature_column_names)
print("Output column:", output_column_name)

#now we can split data as features and outcome
X=df[feature_column_names]
print('features:')
print(X) # test

y=df[output_column_name]
#test
print(y)

#now we split data into training and testing sets with sklearn train_test_split %20 and %80 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#create a decision tree with Train Data
dtree=DecisionTreeClassifier(max_depth=10)
dtree=dtree.fit(X_train,y_train)

#Predict using the test test
y_pred=dtree.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#console option can be activated or deactivated
"""
print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Interpretation
print("\nInterpretation:")
print(f"Accuracy: About {accuracy*100:.2f}% of the predictions were correct.")
print(f"Precision: When the model predicts a positive result (i.e., diabetes), it is correct about {precision*100:.2f}% of the time.")
print(f"Recall: Of all the actual positive cases in the data, the model correctly identified {recall*100:.2f}% of them.")
print(f"F1 Score: The F1 Score is a harmonic mean of Precision and Recall and gives an overall measure of model's performance. An F1 Score closer to 1 indicates better performance.")
"""

#sample prediction

sample_data = X_test.iloc[0]  # taking the first row of the test set as a sample
sample_true_label = y_test.iloc[0]
sample_prediction = dtree.predict([sample_data])

#for test purpose to print console 
print("\nSample Prediction:")
print(f"For the input data: \n{sample_data}")
print(f"True label: {sample_true_label}")
print(f"Predicted label: {sample_prediction[0]}")

if sample_prediction[0] == 0:
    print("The model predicts that this individual does not have diabetes.")
else:
    print("The model predicts that this individual has diabetes.")

#confusion matrix labels are yes or no :1 or 0 
cm=confusion_matrix(y_test,y_pred)

# populate Treeview with dataframe data


def populate_tree(tree, dataframe):
    tree["columns"] = list(dataframe.columns)
    
    #empty columns space
    tree.column("#0", width=4, stretch=tk.NO)   
    
    for col in dataframe.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor=tk.W, width=100)
    
    for index, row in dataframe.iterrows():
        tree.insert("", 0, values=list(row))

# Function to display the decision tree in a new window
def show_decision_tree():
    tree_win = tk.Toplevel(root)
    tree_win.title("Decision Tree")
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(40, 23), dpi=50)  #adjust for your screen
    plot_tree(dtree, feature_names=feature_column_names, filled=True, fontsize=10, ax=ax, proportion=True)
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    tkimg_tree = ImageTk.PhotoImage(img)

    tree_label = ttk.Label(tree_win, image=tkimg_tree)
    tree_label.image = tkimg_tree
    tree_label.pack(padx=20, pady=20)

def predict_new_data():
    #  values from input entries 
    try:
        input_data = [
            float(entries[col].get()) for col in feature_column_names
        ]
    except ValueError:
        result_text.set("Please enter valid numbers!")
        return

    # Predict using the model
    prediction = dtree.predict([input_data])[0]

    # Populate the tree_sample_data with the input data
    tree_sample_data.delete(*tree_sample_data.get_children())
    tree_sample_data.insert("", 0, values=input_data)

    # Display the prediction 
    result_text.set(f"Predicted label: {prediction}\n")
    if prediction == 0:
        result_text.set(result_text.get() + "The model predicts that this individual does not have diabetes.")
    else:
        result_text.set(result_text.get() + "The model predicts that this individual has diabetes.")


def on_quit():
    root.destroy()

def show_about():
    text = """
Context
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective is to predict based on diagnostic measurements whether a patient has diabetes.

Content
Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

Pregnancies: Number of times pregnant
Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressure: Diastolic blood pressure (mm Hg)
SkinThickness: Triceps skin fold thickness (mm)
Insulin: 2-Hour serum insulin (mu U/ml)
BMI: Body mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction: Diabetes pedigree function
Age: Age (years)
Outcome: Class variable (0 or 1)
"""

    about_win = tk.Toplevel(root)
    about_win.geometry("600x400")  # Adjust the window size as per requirement
    about_win.config(bg="lightgray")
    about_win.title("About")
    
    about_text_area = tk.Text(about_win, wrap=tk.WORD, bg="white", fg="black", font=("Arial", 12), relief="flat", borderwidth=1, padx=10, pady=10)
    about_text_area.place(x=5, y=5, width=590, height=390)  
    about_text_area.insert(tk.END, text)
    about_text_area.config(state=tk.DISABLED)  



########Graphic 

# GUI setup
root = tk.Tk()
root.title("Diabetes Prediction Results")
root.geometry("1800x1000")
root.resizable(False,False)

root.config(bg="lightgray")

# Creating a main menu bar
menu = tk.Menu(root)
root.config(menu=menu)

# Creating File menu
file_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Quit", command=on_quit)

# Creating About menu
about_menu = tk.Menu(menu)
menu.add_cascade(label="About", menu=about_menu)
about_menu.add_command(label="About", command=show_about)


# Treeview for Test Data
tree_test_data = ttk.Treeview(root, height=20)

tree_test_data.place(x=10,y=30)

populate_tree(tree_test_data, X_test)
label_test=tk.Label(root,text="Test Data: Source: National Institute of Diabetes and Digestive and Kidney Diseases")
label_test.place(x=10,y=5)

# Treeview for Sample Data
tree_sample_data = ttk.Treeview(root, height=5)

tree_sample_data.place(x=830,y=30)

populate_tree(tree_sample_data, pd.DataFrame([sample_data]))
#label for sample data
label_sample=tk.Label(root,text="Sample Data")
label_sample.place(x=830,y=5)



buf = io.BytesIO()
fig, ax = plt.subplots(figsize=(4,3), dpi=150)  # Increased dpi for better resolution
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
cm_display.plot(ax=ax)
plt.tight_layout()
plt.savefig(buf, format='png')
buf.seek(0)
img = Image.open(buf)
tkimg = ImageTk.PhotoImage(img)

cm_label = ttk.Label(root, image=tkimg)
cm_label.image = tkimg

cm_label.place(x=830,y=200)

# Results Text Widget
results_display = tk.Text(root, height=15, wrap=tk.WORD)

results_display.place(x=10,y=500)
results_display.insert(tk.END, f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
results_display.insert(tk.END, "Interpretation: Accuracy is the ratio of correctly predicted instances to the total instances. A high accuracy indicates that the classifier made fewer mistakes.\n\n")

results_display.insert(tk.END, f"Precision: {precision_score(y_test, y_pred)}\n")
results_display.insert(tk.END, "Interpretation: Precision is the ratio of correctly predicted positive instances to the total predicted positives. A high precision indicates that the classifier returned substantially more relevant results than irrelevant ones.\n\n")

results_display.insert(tk.END, f"Recall: {recall_score(y_test, y_pred)}\n")
results_display.insert(tk.END, "Interpretation: Recall (Sensitivity) measures the proportion of actual positives that were identified correctly. A high recall indicates that the classifier returned most of the relevant results.\n\n")

results_display.insert(tk.END, f"F1 Score: {f1_score(y_test, y_pred)}\n")
results_display.insert(tk.END, "Interpretation: F1 Score is the weighted average of Precision and Recall. It takes both false positives and false negatives into account. A high F1 score indicates a balance between precision and recall.\n\n")

results_display.config(state=tk.DISABLED)

# Button to display decision tree
tree_btn = ttk.Button(root, text="Show Decision Tree", command=show_decision_tree)
tree_btn.place(x=10,y=800)


# Entry  to input data for prediction
entries = {}
for idx, col in enumerate(feature_column_names):
    lbl = ttk.Label(root, text=col,background="lightgray")
    lbl.place(x=670, y=730 + (idx * 30))
    
    ent = ttk.Entry(root)
    ent.place(x=820, y=730 + (idx * 30))
    entries[col] = ent

# Predict button
predict_btn = ttk.Button(root, text="Predict", command=predict_new_data)
predict_btn.place(x=970, y=860 )

# Textbox to display prediction result
result_text = tk.StringVar()
result_display = ttk.Label(root, textvariable=result_text, wraplength=200, anchor="nw", background="white", relief="sunken", padding=10)
result_display.place(x=970, y=450 + (len(feature_column_names) + 1) * 30, width=300, height=120)


root.mainloop()
