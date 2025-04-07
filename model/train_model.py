import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocessing.clean_data import load_and_clean_data

def train_and_save_model():
    df = load_and_clean_data('dataset/sensor_data.csv')
    X = df.drop(columns=['attack', 'timestamp', 'source_address', 'source_id'])
    y = df['attack']

    print("🔎 Class distribution:\n", y.value_counts())

    # Avoid ValueError: if any class has fewer than 2 samples
    if y.value_counts().min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        print("✅ Using stratified split.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("⚠️ Not enough samples to stratify. Using standard split.")

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'model/final_model.pkl')
    print("✅ Model saved!")

if __name__ == '__main__':
    train_and_save_model()
