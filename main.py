import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os

# Vérifier si le fichier modèle existe
model_path = 'breast_cancer_model.h5'
if not os.path.exists(model_path):
    st.error(f"Le fichier modèle '{model_path}' n'existe pas. Assurez-vous qu'il se trouve dans le bon répertoire.")
else:
    # Charger le modèle
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")

    # Titre de l'application
    st.title('Breast Cancer Prediction / Prédiction du Cancer du Sein')

    # Introduction
    st.write("""
    This application uses a deep learning model to predict if a tumor is benign or malignant.
    Enter the characteristics of the tumor to get a prediction.

    Cette application utilise un modèle de deep learning pour prédire si une tumeur est bénigne ou maligne.
    Entrez les caractéristiques de la tumeur pour obtenir une prédiction.
    """)

    # Liste des caractéristiques d'entrée en anglais et français
    feature_names = [
        'mean radius (Rayon moyen)', 'mean texture (Texture moyenne)', 'mean perimeter (Périmètre moyen)',
        'mean area (Surface moyenne)', 'mean smoothness (Lissage moyen)',
        'mean compactness (Compacité moyenne)', 'mean concavity (Concavité moyenne)',
        'mean concave points (Points concaves moyens)', 'mean symmetry (Symétrie moyenne)',
        'mean fractal dimension (Dimension fractale moyenne)',
        'radius error (Erreur du rayon)', 'texture error (Erreur de texture)', 'perimeter error (Erreur de périmètre)',
        'area error (Erreur de surface)', 'smoothness error (Erreur de lissage)',
        'compactness error (Erreur de compacité)', 'concavity error (Erreur de concavité)',
        'concave points error (Erreur des points concaves)', 'symmetry error (Erreur de symétrie)',
        'fractal dimension error (Erreur de dimension fractale)',
        'worst radius (Pire rayon)', 'worst texture (Pire texture)', 'worst perimeter (Pire périmètre)',
        'worst area (Pire surface)', 'worst smoothness (Pire lissage)',
        'worst compactness (Pire compacité)', 'worst concavity (Pire concavité)',
        'worst concave points (Pires points concaves)', 'worst symmetry (Pire symétrie)',
        'worst fractal dimension (Pire dimension fractale)'
    ]

    # Collecte des entrées utilisateur
    input_data = []
    for feature in feature_names:
        value = st.text_input(f'{feature}', value='0.0')
        input_data.append(float(value))

    # Normalisation des données d'entrée
    scaler = StandardScaler()
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.fit_transform(input_data)
    input_data_reshaped = input_data_scaled.reshape(input_data_scaled.shape[0], input_data_scaled.shape[1], 1)

    # Prédiction
    if st.button('Predict / Prédire'):
        try:
            prediction = model.predict(input_data_reshaped)
            predicted_class = np.argmax(prediction, axis=1)

            if predicted_class == 1:
                st.write("The tumor is Malignant / La tumeur est Maligne")
            else:
                st.write("The tumor is Benign / La tumeur est Bénigne")
        except Exception as e:
            st.error(f"Error during prediction: {e} / Erreur lors de la prédiction : {e}")

    # Affichage des informations supplémentaires
    st.write("""
    ### Note
    This application is a decision support tool and does not replace a medical consultation.
    For reliable results, please consult a healthcare professional.

    Cette application est un outil d'aide à la décision et ne remplace pas une consultation médicale.
    Pour des résultats fiables, veuillez consulter un professionnel de la santé.
    """)
