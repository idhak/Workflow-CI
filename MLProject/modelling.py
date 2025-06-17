import pandas as pd
import mlflow
import joblib
import shap
import matplotlib.pyplot as plt
import json
import os
import seaborn as sns
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Memuat data yang sudah diproses
base_dir = os.path.dirname(os.path.abspath(__file__))
X_train = pd.read_csv(os.path.join(base_dir, "data_preprocessing/X_train_processed.csv"))
X_test = pd.read_csv(os.path.join(base_dir, "data_preprocessing/X_test_processed.csv"))
y_train = pd.read_csv(os.path.join(base_dir, "data_preprocessing/y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(base_dir, "data_preprocessing/y_test.csv")).squeeze()

# Mengatur eksperimen MLflow
mlflow.set_experiment("random_forest_experiment_with_tuning")

# Nonaktifkan autolog untuk menangani logging secara manual
mlflow.sklearn.autolog(log_models=False, log_input_examples=False, log_model_signatures=False)

with mlflow.start_run(run_name="Tuned_Random_Forest"):
    # Menyiapkan parameter grid untuk tuning
    search_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': [None, 'balanced']
    }

    # Melakukan grid search dengan cross validation
    model_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=search_params,
        cv=5,
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )
    model_search.fit(X_train, y_train)
    optimal_model = model_search.best_estimator_

    # Membuat prediksi
    predictions = optimal_model.predict(X_test)
    probability_predictions = optimal_model.predict_proba(X_test)[:, 1]

    # Menghitung metrik evaluasi
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probability_predictions)

    # Mencatat metrik ke MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc_score", roc_auc)
    mlflow.log_params(model_search.best_params_)

    # Menyimpan informasi run
    current_run = mlflow.active_run()
    if current_run:
        run_identifier = current_run.info.run_id
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run_identifier)
        mlflow.log_artifact("mlflow_run_id.txt", "run_info")
        os.remove("mlflow_run_id.txt")

        # Menyiapkan contoh input untuk signature
        input_sample = X_train.head(1)
        
        try:
            signature_predictions = optimal_model.predict(input_sample)
            model_signature = infer_signature(input_sample, signature_predictions)
        except Exception as e:
            print(f"Peringatan: Gagal membuat signature model: {e}")
            model_signature = None

        # Mencatat model ke MLflow
        mlflow.sklearn.log_model(
            sk_model=optimal_model,
            artifact_path="optimized_model",
            signature=model_signature,
            input_example=input_sample
        )

    # Menyimpan metrik evaluasi
    evaluation_results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "best_parameters": model_search.best_params_
    }
    
    with open("model_metrics.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    mlflow.log_artifact("model_metrics.json", "performance_metrics")
    os.remove("model_metrics.json")

    # Visualisasi confusion matrix
    try:
        conf_matrix = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.tight_layout()
        conf_matrix_path = "confusion_matrix_plot.png"
        plt.savefig(conf_matrix_path)
        mlflow.log_artifact(conf_matrix_path, "evaluation_visualizations")
        plt.close()
        os.remove(conf_matrix_path)
    except Exception as e:
        print(f"Peringatan: Gagal membuat confusion matrix: {e}")

    # Analisis SHAP
    try:
        print("Menghitung nilai SHAP...")
        explainer = shap.TreeExplainer(optimal_model)
        shap_values = explainer.shap_values(X_test)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=X_train.columns, show=False)
        shap_plot_path = "shap_summary_visualization.png"
        plt.savefig(shap_plot_path, bbox_inches="tight")
        mlflow.log_artifact(shap_plot_path, "feature_importance")
        plt.close()
        os.remove(shap_plot_path)
    except Exception as e:
        print(f"Peringatan: Gagal membuat plot SHAP: {e}")

print("\nProses pelatihan model selesai. Data telah dicatat di MLflow.")
print(f"Hasil Evaluasi - Akurasi: {accuracy:.4f}, Presisi: {precision:.4f}")
print(f"Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
