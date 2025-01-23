import os
import tempfile
from io import BytesIO

from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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



def plot_credit_score_distribution(scores, point, filename):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, color='green', alpha=0.6)
    plt.yscale('log')
    plt.axvline(x=point, color='red', linestyle='--', label=f'Puntaje {point}')
    plt.scatter(point, 1, color='red', s=100, zorder=5)
    plt.title("Distribución de Puntajes de Crédito (300-850)")
    plt.xlabel("Puntaje")
    plt.ylabel("Frecuencia (escala logarítmica)")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()

@app.route("/predict", methods=["POST"])
def predict():
    # # Extract inputs from the form
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

    # # Tamaño de la muestra
    # n_samples = 1000
    #
    # # Generar datos numéricos
    # numeric_data = {
    #     "recoveries": np.random.uniform(0, 5000, n_samples),
    #     "collection_recovery_fee": np.random.uniform(0, 1000, n_samples),
    #     "total_rec_prncp": np.random.uniform(1000, 35000, n_samples),
    #     "out_prncp": np.random.uniform(0, 35000, n_samples),
    #     "last_pymnt_amnt": np.random.uniform(0, 1000, n_samples),
    #     "total_pymnt": np.random.uniform(1000, 40000, n_samples),
    #     "installment": np.random.uniform(100, 1500, n_samples),
    #     "funded_amnt_inv": np.random.uniform(1000, 35000, n_samples),
    #     "loan_amnt": np.random.uniform(1000, 35000, n_samples),
    #     "total_rec_int": np.random.uniform(0, 8000, n_samples),
    #     "total_rec_late_fee": np.random.uniform(0, 100, n_samples),
    #     "int_rate": np.random.uniform(5, 25, n_samples),
    #     "inq_last_6mths": np.random.randint(0, 10, n_samples),
    #     "open_acc": np.random.randint(1, 30, n_samples),
    # }
    #
    # # Generar datos categóricos
    # categorical_data = {
    #     "term": np.random.choice(["36 months", "60 months"], n_samples),
    #     "emp_length": np.random.choice(
    #         [
    #             "< 1 year",
    #             "1 year",
    #             "2 years",
    #             "3 years",
    #             "4 years",
    #             "5 years",
    #             "6 years",
    #             "7 years",
    #             "8 years",
    #             "9 years",
    #             "10+ years",
    #         ],
    #         n_samples,
    #     ),
    #     "home_ownership": np.random.choice(
    #         ["RENT", "OWN", "MORTGAGE", "OTHER"], n_samples
    #     ),
    #     "purpose": np.random.choice(
    #         [
    #             "debt_consolidation",
    #             "credit_card",
    #             "home_improvement",
    #             "small_business",
    #             "major_purchase",
    #             "other",
    #         ],
    #         n_samples,
    #     ),
    #     "grade": np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n_samples),
    #     "initial_list_status": np.random.choice(["w", "f"], n_samples),
    # }

    # Crear DataFrame
    #df = pd.DataFrame({**numeric_data, **categorical_data})
    data_processed = preprocessor.transform(df)
    predictions = model.predict(data_processed).ravel()
    y_pred_proba = (predictions > 0.5).astype(int)
    # Convertir probabilidades a puntajes
    y_scores = np.array([probability_to_score(p) for p in y_pred_proba])
    # Render the result
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plot_credit_score_distribution(y_scores, y_scores[0], temp_file.name)
    temp_file.close()

    # Render the result with the plot
    return render_template("resource.html", prediction=y_scores[0],
                           plot_url=url_for('static', filename=os.path.basename(temp_file.name)))



if __name__ == "__main__":
    app.run(debug=True)
