import joblib
from sklearn.metrics import accuracy_score, precision_score
from preprocessing.clean_data import load_and_clean_data

def evaluate_model():
    model = joblib.load('model/final_model.pkl')
    df = load_and_clean_data('dataset/sensor_data.csv')
    X = df.drop(columns=['attack', 'timestamp', 'source_address', 'source_id'])
    y = df['attack']

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    print(f"🔍 Accuracy: {acc*100:.2f}%")
    print(f"🔍 Precision: {prec*100:.2f}%")

if __name__ == '__main__':
    evaluate_model()
