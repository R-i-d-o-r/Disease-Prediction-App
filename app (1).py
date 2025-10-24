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
    try:
        # Split and clean input symptoms
        input_symptoms = [s.strip().lower() for s in symptom_text.split(",") if s.strip()]
        input_data = [0] * len(symptom_index)

        # Map user symptoms to feature indices
        valid_symptoms = []
        invalid_symptoms = []

        for symptom in input_symptoms:
            if symptom in symptom_index:
                input_data[symptom_index[symptom]] = 1
                valid_symptoms.append(symptom)
            else:
                invalid_symptoms.append(symptom)

        # Handle case when no valid symptoms
        if not valid_symptoms:
            return "⚠️ None of the entered symptoms matched our database. Please check spelling and try again."

        # Create input dataframe
        input_df = pd.DataFrame([input_data], columns=symptoms)

        # Get predictions from models
        rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
        nb_pred = encoder.classes_[nb_model.predict(input_df)[0]]
        svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]
        dt_pred = encoder.classes_[dt_model.predict(input_df)[0]]

        # Find most frequent prediction
        predictions = [rf_pred, nb_pred, svm_pred, dt_pred]
        unique, counts = np.unique(predictions, return_counts=True)
        final_pred = unique[np.argmax(counts)]

        # Build response
        response = (
            f"🌳 Decision Tree: {dt_pred}\n"
            f"🌲 Random Forest: {rf_pred}\n"
            f"⚙️ SVM: {svm_pred}\n"
            f"📈 Naive Bayes: {nb_pred}\n\n"
            f"🩺 Final Prediction: {final_pred}"
        )

        if invalid_symptoms:
            response += f"\n\n⚠️ Ignored unrecognized symptoms: {', '.join(invalid_symptoms)}"

        return response

    except Exception as e:
        return f"❌ An error occurred: {str(e)}"
        

# Gradio Interface
#iface = gr.Interface(
#    fn=predict_disease,
#    inputs=gr.Textbox(lines=2, placeholder="Enter symptoms separated by commas (e.g., fever, cough, fatigue)"),
#    outputs="text",
#    title="AI Disease Prediction App",
#    description="Enter your symptoms and get the predicted disease 🤖 (Built entirely from my phone!)"
#)

#iface.launch(debug=True)
# Fixed symptom list
symptom_list = [
    "fever",
    "headache",
    "nausea",
    "vomiting",
    "fatigue",
    "joint_pain",
    "skin_rash",
    "cough",
    "weight_loss",
    "yellow_eyes"
]

symptom_text_block = (
    "🩺 **Available Symptoms:**\n\n"
    + ", ".join(symptom_list)
    + "\n\n⚠️ Please type symptoms exactly as listed above, separated by commas."
)

# Updated Gradio UI
with gr.Blocks(title="AI Disease Prediction App") as iface:
    gr.Markdown("# 🧠 AI Disease Prediction App")
    gr.Markdown("Enter your symptoms separated by commas (e.g., `fever, cough, fatigue`).\nBuilt entirely from my phone 📱🤖")
    
    with gr.Row():
        input_box = gr.Textbox(
            lines=2,
            placeholder="Enter symptoms here...",
            label="Your Symptoms"
        )
        output_box = gr.Textbox(label="Prediction Result")

    gr.Markdown(symptom_text_block)

    sample_btn = gr.Button("✨ Try Sample Input")
    predict_btn = gr.Button("🔍 Predict Disease")

    sample_btn.click(fn=lambda: "fever, cough, fatigue", inputs=None, outputs=input_box)
    predict_btn.click(predict_disease, inputs=input_box, outputs=output_box)

iface.launch(debug=True)

  
