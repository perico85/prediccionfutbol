from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import requests
from io import StringIO

app = Flask(__name__)

# URLs de los archivos CSV
csv_urls = [
    "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2223/SP1.csv",
    "https://www.football-data.co.uk/mmz4281/2122/SP1.csv"
]

# Función para descargar y leer CSV desde una URL
def read_csv_from_url(url):
    response = requests.get(url)
    return pd.read_csv(StringIO(response.text))

# Combinar todos los archivos CSV
li = []
for url in csv_urls:
    df = read_csv_from_url(url)
    li.append(df)

combined_df = pd.concat(li, axis=0, ignore_index=True)

# Preprocesamiento de datos
le = LabelEncoder()
combined_df['HomeTeam'] = le.fit_transform(combined_df['HomeTeam'])
combined_df['AwayTeam'] = le.fit_transform(combined_df['AwayTeam'])

# Seleccionar características para el modelo
features = ['HomeTeam', 'AwayTeam']
X = combined_df[features]

# Definir las variables objetivo
targets = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
y = combined_df[targets]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelos de Random Forest para cada objetivo
models = {}
for target in targets:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train[target])
    models[target] = rf

# Función para obtener estadísticas y predicciones (modelo IA)
def get_stats_and_predict_ia(home_team, away_team):
    stats = {}

    # Histórico entre los dos equipos
    direct_matches = combined_df[(combined_df['HomeTeam'] == le.transform([home_team])[0]) & (combined_df['AwayTeam'] == le.transform([away_team])[0])]
    stats['Historico'] = direct_matches[targets].mean().to_dict()
    
    total_matches = len(direct_matches)
    home_wins = sum(direct_matches['FTR'] == 'H')
    away_wins = sum(direct_matches['FTR'] == 'A')
    draws = sum(direct_matches['FTR'] == 'D')
    stats['Historico']['Results'] = f"Victorias local: {home_wins}, Victorias visitante: {away_wins}, Empates: {draws}"

    # Estadísticas en casa del equipo local
    home_stats = combined_df[combined_df['HomeTeam'] == le.transform([home_team])[0]]
    stats['Home'] = home_stats[targets].mean().to_dict()

    # Estadísticas fuera del equipo visitante
    away_stats = combined_df[combined_df['AwayTeam'] == le.transform([away_team])[0]]
    stats['Away'] = away_stats[targets].mean().to_dict()

    # Otros equipos jugando en casa contra el equipo visitante
    other_home_stats = combined_df[(combined_df['AwayTeam'] == le.transform([away_team])[0]) & (combined_df['HomeTeam'] != le.transform([home_team])[0])]
    stats['OtherHome'] = other_home_stats[targets].mean().to_dict()

    # Otros equipos jugando fuera contra el equipo local
    other_away_stats = combined_df[(combined_df['HomeTeam'] == le.transform([home_team])[0]) & (combined_df['AwayTeam'] != le.transform([away_team])[0])]
    stats['OtherAway'] = other_away_stats[targets].mean().to_dict()

    # Predicción
    home_encoded = le.transform([home_team])[0]
    away_encoded = le.transform([away_team])[0]
    
    prediction_features = [home_encoded, away_encoded]
    
    predictions = {}
    for target in targets:
        pred = models[target].predict([prediction_features])[0]
        predictions[target] = round(pred, 2)

    stats['Prediccion'] = predictions

    return stats

# Función para obtener estadísticas (modelo estadístico)
def get_stats(home_team, away_team):
    stats = {}

    columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']

    # Histórico entre los dos equipos
    direct_matches = combined_df[(combined_df['HomeTeam'] == le.transform([home_team])[0]) & (combined_df['AwayTeam'] == le.transform([away_team])[0])]
    stats['Historico'] = direct_matches[columns].mean().to_dict()
    
    total_matches = len(direct_matches)
    home_wins = sum(direct_matches['FTR'] == 'H')
    away_wins = sum(direct_matches['FTR'] == 'A')
    draws = sum(direct_matches['FTR'] == 'D')
    stats['Historico']['Results'] = f"Victorias local: {home_wins}, Victorias visitante: {away_wins}, Empates: {draws}"

    # Estadísticas en casa del equipo local
    home_stats = combined_df[combined_df['HomeTeam'] == le.transform([home_team])[0]]
    stats['Home'] = home_stats[columns].mean().to_dict()

    # Estadísticas fuera del equipo visitante
    away_stats = combined_df[combined_df['AwayTeam'] == le.transform([away_team])[0]]
    stats['Away'] = away_stats[columns].mean().to_dict()

    # Otros equipos jugando en casa contra el equipo visitante
    other_home_stats = combined_df[(combined_df['AwayTeam'] == le.transform([away_team])[0]) & (combined_df['HomeTeam'] != le.transform([home_team])[0])]
    stats['OtherHome'] = other_home_stats[columns].mean().to_dict()

    # Otros equipos jugando fuera contra el equipo local
    other_away_stats = combined_df[(combined_df['HomeTeam'] == le.transform([home_team])[0]) & (combined_df['AwayTeam'] != le.transform([away_team])[0])]
    stats['OtherAway'] = other_away_stats[columns].mean().to_dict()

    return stats

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    home_team = data['home_team']
    away_team = data['away_team']
    model = data.get('model', 'ia')  # 'ia' o 'estadistico'
    
    if model == 'ia':
        results = get_stats_and_predict_ia(home_team, away_team)
    else:
        results = get_stats(home_team, away_team)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
