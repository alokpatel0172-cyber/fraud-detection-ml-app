from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Model aur columns load karo
model   = pickle.load(open('model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

print(" Model loaded! Expects", model.n_features_in_, "features")
print(" Columns:", columns)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Form se data lo
        data = {
            'Transaction_Amount':     float(request.form['amount']),
            'Transaction_Frequency':  float(request.form['frequency']),
            'Avg_Transaction_Amount': float(request.form['avg_amount']),
            'Transaction_Type':       request.form['transaction_type'],
            'Payment_Gateway':        request.form['payment_gateway'],
            'Device_OS':              request.form['device_os'],
            'Transaction_City':       request.form['city'],
            'Transaction_State':      request.form['state'],
            'Transaction_Status':     request.form['status'],
            'Merchant_Category':      request.form['merchant_category'],
            'Transaction_Channel':    request.form['channel'],
        }

        df = pd.DataFrame([data])

        # Categorical columns encode karo
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))

        # Exact same column order use karo jo training mein thi
        df = df.reindex(columns=columns, fill_value=0)

        print("Input shape:", df.shape)
        print("Columns:", list(df.columns))

        prediction  = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        confidence  = round(max(probability) * 100, 2)
        result      = "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE Transaction"
        color       = "danger" if prediction == 1 else "success"

        return render_template('result.html',
                               result=result,
                               confidence=confidence,
                               color=color)

    except Exception as e:
        return render_template('result.html',
                               result=f"Error: {str(e)}",
                               confidence=0,
                               color="warning")

if __name__ == '__main__':
    app.run(debug=True)
