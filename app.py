import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

# this will be the starting point from which our flask app will run
app = Flask(__name__)

# loading the model
model = pickle.load(open('regmodel.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html') # whenever user goes to the / page, this will redirect the user to the home.html page
                                        # home.html is in the templates folder

@app.route('/predict_api', methods = ['POST']) # we use this so that we can send the request to our website and get the output

def predict_api():
    data = request.json['data'] # the Json data we give to the predict_api will be captured and stored in data variable
    new_data = np.array(list(data.values())).reshape(1, -1) # as the data is in json format, we put the values in a list and use reshape to 
                                                            # convert it to a 2D array (as the model expects)

    output = model.predict(new_data) # here we pass the list to the model
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    output = model.predict(final_input)
    return render_template("home.html", prediction_text = "The predicted rent is {}".format(output)) # prediction_text placeholder wil 
                                                                                                     # be replaced with output

if __name__ == "__main__": # we use this to run the code
    app.run(debug=True) 