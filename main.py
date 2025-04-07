from model.train_model import train_and_save_model
from evaluation.evaluate_model import evaluate_model

if __name__ == '__main__':
    print("📦 Training Model...")
    train_and_save_model()
    print("📈 Evaluating Model...")
    evaluate_model()
