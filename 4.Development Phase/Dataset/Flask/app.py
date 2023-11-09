from flask import Flask, render_template, request
import pickle

app = Flask(__name__, static_url_path='/static')

with open('disease_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    symptoms_dict = request.form.to_dict()
    columns=['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
       'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
       'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
       'spotting_ urination', 'fatigue', 'weight_loss', 'restlessness',
       'lethargy', 'patches_in_throat', 'cough', 'high_fever', 'sunken_eyes',
       'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
       'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
       'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain',
       'diarrhoea', 'mild_fever', 'yellowing_of_eyes', 'swelled_lymph_nodes',
       'malaise', 'blurred_and_distorted_vision', 'phlegm', 'congestion',
       'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
       'irritation_in_anus', 'neck_pain', 'dizziness', 'obesity',
       'swollen_legs', 'puffy_face_and_eyes', 'excessive_hunger',
       'extra_marital_contacts', 'knee_pain', 'muscle_weakness', 'stiff_neck',
       'swelling_joints', 'movement_stiffness', 'loss_of_balance',
       'unsteadiness', 'weakness_of_one_body_side', 'bladder_discomfort',
       'foul_smell_of urine', 'passage_of_gases', 'depression', 'irritability',
       'muscle_pain', 'red_spots_over_body', 'belly_pain',
       'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
       'increased_appetite', 'family_history', 'mucoid_sputum', 'rusty_sputum',
       'lack_of_concentration', 'visual_disturbances',
       'receiving_blood_transfusion', 'coma', 'history_of_alcohol_consumption',
       'blood_in_sputum', 'palpitations', 'pus_filled_pimples', 'blackheads',
       'scurring', 'inflammatory_nails', 'yellow_crust_ooze']
    chosen_symptoms=symptoms_dict.keys()
    values=[]
    for symptom in columns:
        if symptom in chosen_symptoms:
            values.append(1)
        else:
            values.append(0)
    prediction = model.predict([values])
    return render_template("index.html", prediction="You might be having "+prediction[0]+", please visit a doctor, stay safe")

if __name__ == "__main__":
    app.run(debug=True)
