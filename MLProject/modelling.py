import mlflow
import pandas as pd
import os
import numpy as np
import warnings
import sys
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dotenv import load_dotenv
import dagshub.auth

def train_model(train_data, test_data, target_name, rf_params):
    # Membuat Folder untuk artifact
    os.makedirs(f"./artifact", exist_ok=True)

    # Memisah fitur dan target
    x_train = train_data.drop(target_name, axis=1)
    y_train = train_data[target_name]
    x_test = test_data.drop(target_name, axis=1)
    y_test = test_data[target_name]

    with mlflow.start_run():
        # Pembuatan Objek Random Forest
        model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params['min_samples_leaf']
        )

        model.fit(x_train, y_train)
        dump(model, "artifact/model.pkl")
        
        mlflow.sklearn.log_model(
            sk_model=model,
            input_example=x_train[0:5],
            registered_model_name=None,
            artifact_path="model",
        )

        # Log Important Model
        feature_importances = pd.DataFrame({
            'feature': x_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig("artifact/feature_importances.png")
        plt.close()

        # Evaluate the model on the test set and log accuracy
        y_pred_train = model.predict(x_train)
        y_pred_test = model.predict(x_test)

        # Buat dan simpan ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
        plt.title("ROC Curve")
        plt.savefig("artifact/roc_curve.png")
        plt.close(fig)

        # Log COnfusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix")
        plt.savefig("artifact/confusion_matrix.png")
        plt.close(fig)
        

if __name__ == "__main__":
    load_dotenv()
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Define Variable
    train_path = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "aids_preprocessing/train.csv")
    test_path  = sys.argv[6] if len(sys.argv) > 6 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "aids_preprocessing/test.csv")
    target_column = "infected"

    # Load Dataset
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Parameter Grid untuk bayesian search
    params = {
        'n_estimators': int(sys.argv[1]) if len(sys.argv) > 1 else 300,
        'max_depth': int(sys.argv[2]) if len(sys.argv) > 2 else 15,
        'min_samples_split': int(sys.argv[3]) if len(sys.argv) > 3 else 5,
        'min_samples_leaf': int(sys.argv[4]) if len(sys.argv) > 4 else 1
    }

    # Training Model
    train_model(train, test, target_column, params)
