import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
modellcd = pickle.load(open('modellcd.pkl','rb'))

@app.route('/',methods=["GET", "POST"])
def home1():
  return render_template('home1.html')

@app.route('/diseases',methods=["GET", "POST"])
def diseases():
  return render_template('diseases.html')

@app.route('/hospitals',methods=["GET", "POST"])
def hospitals():
  return render_template('hospitals.html')

@app.route('/index',methods=["GET", "POST"])
def index():
  return render_template('index.html')

@app.route('/lcd',methods=["GET", "POST"])
def lcd():
  return render_template('lcd.html')

@app.route('/index11',methods=["GET", "POST"])
def index11():
  return render_template('index11.html')


@app.route('/licindex',methods=["GET", "POST"])
def licindex():
  return render_template('licindex.html')



@app.route('/chat',methods=["GET", "POST"])
def chat():
  return render_template('chat.html')


@app.route('/predict',methods=["GET", "POST"])
def predict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model.predict(df)

  if output == 4:
      res_val = "Bengin Breast cancer"
  else:
      res_val = "Maligant Breast cancer"


  return render_template('index.html', prediction_text='Patient might have {}'.format(res_val))


@app.route('/predictblood',methods=["GET", "POST"])
def predictblood():
    gender = request.form.get('GENDER')
    age = request.form.get('AGE')
    smoking = request.form.get('SMOKING')
    yellow_fingers = request.form.get('YELLOW_FINGERS')
    anxiety = request.form.get('ANXIETY')
    peer_pressure = request.form.get('PEER_PRESSURE')
    chronic_disease = request.form.get('CHRONIC_DISEASE')
    fatigue = request.form.get('FATIGUE')
    allergy = request.form.get('ALLERGY')
    wheezing = request.form.get('WHEEZING')
    alcohol_consuming = request.form.get('ALCOHOL_CONSUMING')
    coughing = request.form.get('COUGHING')
    shortness_of_breath = request.form.get('SHORTNESS_OF_BREATH')
    swallowing_difficulty = request.form.get('SWALLOWING_DIFFICULTY')
    chest_pain = request.form.get('CHEST_PAIN')

    output1 = modellcd.predict([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                             allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty,
                             chest_pain]])

    if output1 == 1:
        res_val = "Lung cancer"
    else:
        res_val = "Not Lung cancer"

    return render_template('lcd.html', prediction_text='Patient might have {}'.format(res_val))


if __name__ == "__main__":
    app.run(debug=True)