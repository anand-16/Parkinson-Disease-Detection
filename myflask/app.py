from flask import Flask, render_template, request
from flask import jsonify
import requests
import pickle
import numpy as np
import sklearn

app = Flask(__name__)
model = pickle.load(open('care.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        mdvp_fo=float(request.form['mdvp_fo'])
        mdvp_fhi=float(request.form['mdvp_fhi'])
        mdvp_flo=float(request.form['mdvp_flo'])
        mdvp_jitter1=float(request.form['mdvp_jitter1'])
        mdvp_jitter2=float(request.form['mdvp_jitter2'])
        mdvp_rap=float(request.form['mdvp_rap'])
        mdvp_ppq=float(request.form['mdvp_ppq'])
        jitter_ddp=float(request.form['jitter_ddp'])
        mdvp_shimmer=float(request.form['mdvp_shimmer'])
        mdvp_shimmer2=float(request.form['mdvp_shimmer2'])
        shimmer_apq3=float(request.form['shimmer_apq3'])
        shimmer_apq5=float(request.form['shimmer_apq5'])
        mdvp_apq=float(request.form['mdvp_apq'])
        shimmer_dda=float(request.form['shimmer_dda'])
        nhr=float(request.form['nhr'])
        hnr=float(request.form['hnr'])
        rpde=float(request.form['rpde'])
        d2=float(request.form['d2'])
        dfa=float(request.form['dfa'])
        spread1=float(request.form['spread1'])
        spread2=float(request.form['spread2'])
        ppe=float(request.form['ppe'])
        
        features=np.array([mdvp_fo,mdvp_fhi,mdvp_flo,mdvp_jitter1,mdvp_jitter2,mdvp_rap,mdvp_ppq,jitter_ddp,mdvp_shimmer,mdvp_shimmer2,shimmer_apq3,shimmer_apq5,mdvp_apq,shimmer_dda,nhr,hnr,rpde,d2,dfa,spread1,spread2,ppe]) 

        prediction = model.predict([features])
        print(prediction)
        
        if (prediction[0] == 0):
            return render_template('index.html',prediction_text="The Person does not have Parkinsons Disease")
        
        else:
            return render_template('index.html', prediction_text="The Person has Parkinsons")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)