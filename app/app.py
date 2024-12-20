import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

# Charger les fichiers pickle
try:
    with open("config.pickle", "rb") as f:
        config_dict = pickle.load(f)

    with open("model.pickle", "rb") as f:
        model = pickle.load(f)

    with open("Polynomialfeature.pickle", "rb") as f:
        poly_features = pickle.load(f)

    with open("pipeline.pickle", "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Erreur de chargement des fichiers pickle : {e}")
    st.stop()

# Interface utilisateur avec Streamlit
st.title("Prédiction des prix")

# Widgets pour les inputs utilisateur
make = st.selectbox("Marque", config_dict.get("marque", ["N/A"]))
model_name = st.selectbox("Modèle", config_dict.get("modele", ["N/A"]))

year = st.slider(
    "Année",
    1990,
    2024,
    step=1
)

condition = st.slider(
    "Note estimée",
    config_dict.get("note_estimée", [0, 100])[0],
    config_dict.get("note_estimée", [0, 100])[1],
    step=1
)

odometer = st.number_input(
    "Kilométrage",
    min_value=config_dict.get("kilometrage", [0, 500000])[0],
    max_value=config_dict.get("kilometrage", [0, 500000])[1],
    step=1000
)

if st.button("Prédire"):
    # Préparer les données pour le modèle
    input_data = {
        "marque": make,
        "modele": model_name,
        "annee": year,
        "note_estimée": condition,
        "kilometrage": odometer
    }
    st.write(input_data)
    # Créer un DataFrame pour une seule ligne
    input_df = pd.DataFrame([input_data])

    try:
        # Étape 1 : Transformation initiale avec le pipeline
        input_transformed = pipeline.transform(input_df)
        
        # Étape 2 : Application de la transformation polynomiale
        input_poly = poly_features.transform(input_transformed)

        # Vérifier la compatibilité des dimensions
        expected_features = model.coef_.shape[0] if hasattr(model, "coef_") else input_poly.shape[1]
        if input_poly.shape[1] != expected_features:
            st.error(f"Erreur : Le modèle attend {expected_features} caractéristiques, mais {input_poly.shape[1]} ont été fournies.")
        else:
            # Étape 3 : Faire une prédiction
            prediction = model.predict(input_poly)
            st.success(f"Le prix prédit est : {prediction[0]:,.2f} €")
    except ValueError as e:
        st.error(f"Erreur de compatibilité des données : {e}")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors de la transformation des données ou de la prédiction : {e}")

