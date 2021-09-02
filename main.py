from flask import Flask,request,render_template
from test import FirstLinearModel
app = Flask(__name__)


@app.route('/',methods = ['GET','POST'])
def ind():
    return render_template('index.html')


@app.route('/predict',methods = ['GET','POST'])
def predict():
    Air_temperature = request.form['txtAirTemp']
    Process_temperature = request.form['txtProcessTemp']
    Rotational_speed = request.form['txtRationalSpeed']
    Torque = request.form['txtTroque']
    Tool_wear = request.form['txtToolware']
    TWF = request.form['txtTwf']
    HDF = request.form['txtHdf']
    PWF = request.form['txtPwf']
    OSF = request.form['txtOsf']
    RNF = request.form['txtRns']
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])

    test1 = obj.test_transform([[Air_temperature,Process_temperature,Rotational_speed,Torque,Tool_wear,
                                 TWF,HDF,PWF,OSF,RNF]])
    predict = obj.predict(test1)
    return render_template('predict.html',post = predict[0])

@app.route('/score')
def score():
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])
    score=  obj.score()
    return render_template('linearModelS.html',post = score)

@app.route('/Lasso_score')
def lassoscore():
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])
    score = obj.lasso_score()
    return render_template('lassoS.html',post = score)

@app.route('/Ridge_score')
def ridgescore():
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])
    score = obj.Ridge_score()


@app.route('/ElasticNet_score')
def elasticscore():
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])
    score = obj.ElasticNEt_score()
    return render_template('elast.html',post = score)


@app.route('/Multicolinearity')
def mul():
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])
    mul = obj.Multicolinearity()
    return render_template('simple.html', tables=[mul.to_html(classes='data')] ,titles=mul.columns.values)

@app.route('/MyObservations')
def obs():

    return render_template('observation.html')

@app.route('/Report')
def rep():
    '''
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])
    obj.Report().to_file('Report.html')
    '''
    return render_template('Report.html')

@app.route('/adjusted_r2')
def adj():
    obj = FirstLinearModel("C:/Users/Panda/Downloads/ai4i2020.csv")
    obj.Y(['Machine failure'])
    obj.X(['UDI', 'Product ID', 'Type', 'Machine failure'])
    r2 = obj.adjusted_r_square(obj.x_test(), obj.y_test())
    return render_template('adjustedr2.html',post = r2)

if __name__ == '__main__':
    app.run()