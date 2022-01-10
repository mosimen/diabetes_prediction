
import pickle
import numpy as np
from flask import Flask,request,render_template
import xgboost

app=Flask(__name__)
model= open("C:/users/daniel/documents/ds/diabetes_prediction/diabetes_prediction.pkl","rb")
prediction=pickle.load(model)



@app.route("/")
def home():
    return render_template("home.html")

#'num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin',
#       'bmi', 'diab_pred', 'age', 'skin'

@app.route("/predict",methods=["GET","POST"])
def predict():
    
    if request.method=="POST":
        
        num_preg=int(request.form["num_preg"])
        glucose_conc=int(request.form["glucose_conc"])
        diastolic_bp=int(request.form["diastolic_bp"])
        thickness=int(request.form["thickness"])
        insulin=float(request.form["insulin"])
        bmi=float(request.form["bmi"])
        diab_pred=float(request.form["diab_pred"])
        age=int(request.form["age"])
        skin=float(request.form["skin"])
        
        output=prediction.predict([[num_preg, glucose_conc, diastolic_bp, thickness, insulin,bmi, diab_pred, age, skin]])
        prob=prediction.predict_proba([[num_preg, glucose_conc, diastolic_bp, thickness, insulin,bmi, diab_pred, age, skin]])
        
        #prob=round(prob[0],2)
        if output==1:
            result1="Patient is Diabetic"
            result2="Probability is {}".format((round(prob[:,1][0],2)*100))
        else:
            result1="Patient is Not Diabetic"
            result2="Probability is {}".format((round(prob[:,0][0],2)*100))
        
        return render_template("home.html", prediction_text1=result1, prediction_text2=result2)
    
    return render_template("home.html")

if __name__=="__main__":
    app.run(debug=True)