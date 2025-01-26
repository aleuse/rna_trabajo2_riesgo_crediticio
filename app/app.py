import os
import tempfile
import base64

from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from utils.create_and_save_plot import plot_credit_score_distribution

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("models/loan_model_2.h5")
preprocessor = joblib.load("models/preprocessor_2.pkl")


# Función para convertir probabilidades a puntajes
def probability_to_score(prob, base_score=300, max_score=850, pdo=50):
    """
    Convierte probabilidades en puntajes de crédito dentro del rango [base_score, max_score].
    Args:
        prob (float): Probabilidad de default.
        base_score (int): Puntaje base (ej. 300).
        max_score (int): Puntaje máximo (ej. 850).
        pdo (int): Puntos para doblar las probabilidades (por defecto 50).
    Returns:
        score (float): Puntaje de crédito ajustado al rango.
    """
    odds = (1 - prob) / prob
    factor = pdo / np.log(2)
    offset = base_score + 200  # Ajustar el offset para centrar el rango en [300, 850]
    raw_score = offset - factor * np.log(odds)

    # Normalizar el score dentro del rango [300, 850]
    score = np.clip(raw_score, base_score, max_score)
    return score




# Define routes
@app.route("/")
def home():
    return render_template("index.html")  # Renders the form


@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form
    recoveries = float(input_data["recoveries"])
    collection_recovery_fee = float(input_data["collection_recovery_fee"])
    total_rec_prncp = float(input_data["total_rec_prncp"])
    out_prncp = float(input_data["out_prncp"])
    last_pymnt_amnt = float(input_data["last_pymnt_amnt"])
    total_pymnt = float(input_data["total_pymnt"])
    installment = float(input_data["installment"])
    funded_amnt_inv = float(input_data["funded_amnt_inv"])
    loan_amnt = float(input_data["loan_amnt"])
    total_rec_int = float(input_data["total_rec_int"])
    total_rec_late_fee = float(input_data["total_rec_late_fee"])
    int_rate = float(input_data["int_rate"])
    inq_last_6mths = int(input_data["inq_last_6mths"])
    open_acc = int(input_data["open_acc"])

    term = str(input_data["term"])
    emp_length = str(input_data["emp_length"])
    home_ownership = str(input_data["home_ownership"])
    purpose = str(input_data["purpose"])
    grade = str(input_data["grade"])
    initial_list_status = str(input_data["initial_list_status"])

    # Convert inputs to a NumPy array for prediction
    input_array = np.array(
        [
            [
                recoveries,
                collection_recovery_fee,
                total_rec_prncp,
                out_prncp,
                last_pymnt_amnt,
                total_pymnt,
                installment,
                funded_amnt_inv,
                loan_amnt,
                total_rec_int,
                total_rec_late_fee,
                int_rate,
                inq_last_6mths,
                open_acc,
                term,
                emp_length,
                home_ownership,
                purpose,
                grade,
                initial_list_status,
            ]
        ]
    )

    columns = [
        "recoveries",
        "collection_recovery_fee",
        "total_rec_prncp",
        "out_prncp",
        "last_pymnt_amnt",
        "total_pymnt",
        "installment",
        "funded_amnt_inv",
        "loan_amnt",
        "total_rec_int",
        "total_rec_late_fee",
        "int_rate",
        "inq_last_6mths",
        "open_acc",
        "term",
        "emp_length",
        "home_ownership",
        "purpose",
        "grade",
        "initial_list_status",
    ]

    df = pd.DataFrame(input_array, columns=columns)

    data_processed = preprocessor.transform(df)
    predictions = model.predict(data_processed).ravel()
    y_pred_proba = (predictions > 0.5).astype(int)

    y_scores = np.array([probability_to_score(p) for p in y_pred_proba])
    buffer = plot_credit_score_distribution(y_scores, y_scores[0])  # Puedes pasar datos personalizados si es necesario
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return render_template("resource.html", prediction=y_scores[0],
                           plot_url=f"data:image/png;base64,{plot_data}")




if __name__ == "__main__":
    app.run(debug=True)
