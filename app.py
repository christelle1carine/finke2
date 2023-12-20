

'''from flask import Flask, request, jsonify
import numpy as np
import sklearn
import pickle

# Charger le modèle préalablement enregistré
with open('regression_model_saved.pkl', 'rb') as file:
    model = pickle.load(file)

# Créer une instance de l'application Flask
app = Flask(__name__)

# Définir une route pour l'API
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Obtenir les données du billet à partir de la requête JSON
        data = request.get_json()

        # Prétraiter les données et les convertir en tableau numpy
        features = np.array(data['features']).reshape(1, -1)

        # Faire une prédiction avec le modèle
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        # Préparer la réponse JSON
        response = {
            'message': 'Vrai billet' if prediction[0] == 1 else 'Faux billet',
            'proba': float(probability[0, 1])
        }

        # Renvoyer la réponse JSON
        return jsonify(response)
    else:
        return 'Méthode non autorisée. Utilisez la méthode POST pour cette route.'

# Lancer l'application Flask
if __name__ == '__main__':
    app.run()'''
    
    
from flask import Flask, request, jsonify
import numpy as np
import sklearn
import pickle

# Charger le modèle préalablement enregistré
with open('regression_model_saved.pkl', 'rb') as file:
    model = pickle.load(file)

# Créer une instance de l'application Flask
app = Flask(__name__)

# Définir une route pour l'API

@app.route("/")
def une_fonction():
    return "bonjour"

@app.route('/predict', methods=['POST'])
def predict():
    # Obtenir les données du billet à partir de la requête JSON
    data = request.get_json()

    # Prétraiter les données et les convertir en tableau numpy
    features = np.array([
        data['diagonal'],
        data['height_left'],
        data['height_right'],
        data['margin_up'],
        data['length'],
        data['margin_low']
    ]).reshape(1, -1)

    # Faire une prédiction avec le modèle
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    # Préparer la réponse JSON
    response = {
        'message': 'Vrai billet' if prediction[0] == 1 else 'Faux billet',
        'proba': float(probability[0, 1])
    }

    # Renvoyer la réponse JSON
    return jsonify(response)

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)