from flask import Flask,render_template,request,url_for
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/iris'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class t1(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    sepal_lenght = db.Column(db.Integer,nullable=False)
    sepal_width = db.Column(db.Integer,nullable=False)
    petal_lenght = db.Column(db.Integer,nullable=False)
    petal_width = db.Column(db.Integer,nullable=False)
    pred = db.Column(db.String(50),nullable=False)

f1 = open('enc','rb')
f2 = open('sclr','rb')
f3 = open('clf','rb')

enc = pickle.load(f1)
sclr = pickle.load(f2)
clf = pickle.load(f3)

@app.route("/",methods=['GET'])
def index():

    return render_template('index.html')

@app.route("/predict",methods=['POST','GET'])
def predict():
    vals = [float(x) for x in request.form.values()]
    arr = np.array([vals])
    scled = sclr.transform(arr)
    pred = clf.predict(scled)

    entry = t1(sepal_lenght=vals[0],sepal_width=vals[1], petal_lenght=vals[2],petal_width=vals[3], pred=enc.inverse_transform(pred)[0])
    db.session.add(entry)
    db.session.commit()

    return render_template('index.html',preds=enc.inverse_transform(pred)[0].upper(),vals=vals,scled=scled,pred=pred)

if __name__ == "__main__":
    app.run(debug=True)
