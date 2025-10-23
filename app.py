import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import gradio as gr

# Load dataset
data = pd.read_csv("improved_disease_dataset.csv")

# Encode target labels
encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

# Features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Balance dataset
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Train models
dt_model = DecisionTreeClassifier(random_state=42).fit(X_resampled, y_resampled)
rf_model = RandomForestClassifier(random_state=42).fit(X_resampled, y_resampled)
svm_model = SVC().fit(X_resampled, y_resampled)
nb_model = GaussianNB().fit(X_resampled, y_resampled)

# Map symptoms to indices
symptoms = X.columns.values
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

# Prediction function
def predict_disease(symptom_text):
    input_symptoms = symptom_text.split(",")
    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        symptom = symptom.strip()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_df = pd.DataFrame([input_data], columns=symptoms)

    rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_df)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]
    dt_pred = encoder.classes_[dt_model.predict(input_df)[0]]
    final_pred = mode([rf_pred, nb_pred, svm_pred, dt_pred])[0][0]

    return (
        f"🌳 Decision Tree: {dt_pred}\n"
        f"🌲 Random Forest: {rf_pred}\n"
        f"⚙️ SVM: {svm_pred}\n"
        f"📈 Naive Bayes: {nb_pred}\n\n"
        f"🩺 Final Prediction: {final_pred}"
    )

# Gradio Interface
iface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Textbox(lines=2, placeholder="Enter symptoms separated by commas (e.g., fever, cough, fatigue)"),
    outputs="text",
    title="AI Disease Prediction App",
    description="Enter your symptoms and get the predicted disease 🤖 (Built entirely from my phone!)"
)

iface.launch()
  
