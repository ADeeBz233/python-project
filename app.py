import keras
import numpy as np
from flask import Flask,request,jsonify,render_template
app = Flask(__name__)
model = keras.models.load_model('annmodel', compile = False)
@app.route('/')
def home():
  return render_template('api.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [[float(x) for x in request.form.values()]]
    final_features = [[np.array(int_features)]]
    prediction = model.predict(int_features)

    return render_template('api.html',prediction_text='chance for diabetes : {} '.format(prediction))

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)