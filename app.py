'''
Author: Bhaskar Singh
Date: 28th September, 2021
Email: rougesocket@gmail.com
'''


from flask import Flask,render_template,request,redirect,url_for
import numpy as np
import joblib

model=joblib.load('Model.pkl')

app = Flask(__name__)

def format_result(value):

    if value==0:
        return {
        "title":"Not a defaulter",
        "img_value":"1",
        "text_class_color":"text-success",
        "icon_class":"fa-check-circle",
        "description":"The details entered corresponds to sign which most of the Non-defaulter show."}
    else:
        return {
            "title": "defaulter",
            "img_value": "0",
            "text_class_color": "text-danger",
            "icon_class": "fa-times-circle",
            "description": "The details entered corresponds to sign which most of the defaulter show."}


@app.route("/index")
def index():
    #returning the homepage
    return render_template('index.html')


@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        #getting input from the form and typecasting in appropriate form
        income= int(request.form['ip1'])
        current_job_years = int(request.form['ip2'])
        experience = int(request.form['ip3'])
        ip = np.array([income,current_job_years,experience],ndmin=2)
        output= model.predict(ip)
        ans=format_result(output)
        return render_template("predict.html",ans=ans)
    else:
        #redirecting the user to the index
        return redirect(url_for("index"))


if __name__=='__main__': # for if we run outside script directly
    app.run(debug=True)