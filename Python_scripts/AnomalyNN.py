# Importer les bibliothèques requises
import torch  # Bibliothèque pour l'apprentissage profond
import torch.nn as nn  # Pour les composants des réseaux de neurones
import torch.optim as optim  # Pour les algorithmes d'optimisation
import pandas as pd  # Pour la manipulation et l'analyse de données
from sklearn.model_selection import train_test_split  # Pour diviser le jeu de données
from sklearn.preprocessing import MinMaxScaler  # Pour normaliser les caractéristiques

# Charger le jeu de données à partir d'un fichier CSV (remplacez 'data.csv' par votre chemin de fichier)
data = pd.read_csv("data.csv")

# Séparer les caractéristiques (X) et les étiquettes (y)
X = data[['T_Val', 'I_Val', 'U_Val']].values  # Colonnes des caractéristiques
# Étiquettes indiquant les anomalies pour la température, le courant et la tension
y = data[['T_Anomaly', 'I_Anomaly', 'U_Anomaly']].values

# Normaliser les caractéristiques pour les mettre à l'échelle entre 0 et 1 pour une meilleure performance de formation
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Diviser le jeu de données en sous-ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convertir les données en tenseurs PyTorch pour l'entrée du modèle
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Définir l'architecture du réseau neuronal
class AnomalyNN(nn.Module):
    def __init__(self):
        super(AnomalyNN, self).__init__()
        # Définition des couches du réseau
        self.fc1 = nn.Linear(3, 64)  # Première couche : 3 entrées (caractéristiques), 64 neurones
        self.fc2 = nn.Linear(64, 32)  # Deuxième couche : 64 entrées, 32 neurones
        self.fc3 = nn.Linear(32, 16)  # Troisième couche : 32 entrées, 16 neurones
        self.fc4 = nn.Linear(16, 8)  # Quatrième couche : 16 entrées, 8 neurones
        self.fc5 = nn.Linear(8, 3)   # Couche de sortie : 8 entrées, 3 sorties (une pour chaque anomalie)

    def forward(self, x):
        # Appliquer la fonction d'activation ReLU pour chaque couche cachée
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)  # Couche de sortie (aucune activation, utilisée avec BCEWithLogitsLoss)
        return x

# Instancier le modèle de réseau neuronal
model = AnomalyNN()

# Définir la fonction de perte avec une option pour gérer le déséquilibre des classes
# Les poids peuvent être ajustés si certaines classes sont sous-représentées
pos_weights = torch.tensor([1.0, 1.0, 1.0])  # Poids pour équilibrer les classes si nécessaire
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

# Définir l'optimiseur (Adam) et fixer le taux d'apprentissage
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Définir un planificateur de taux d'apprentissage pour réduire le taux en cas de stagnation
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Boucle d'entraînement avec arrêt anticipé et ajustement du taux d'apprentissage
patience = 10  # Nombre maximum d'époques sans amélioration
best_loss = float('inf')  # Meilleure perte initialisée à l'infini
patience_counter = 0  # Compteur pour l'arrêt anticipé

num_epochs = 1000  # Nombre maximum d'époques
loss_values = []  # Liste pour stocker les valeurs de perte
for epoch in range(num_epochs):
    model.train()  # Mettre le modèle en mode entraînement
    optimizer.zero_grad()  # Effacer les gradients de l'étape précédente
    logits = model(X_train)  # Passage avant : calculer la sortie du modèle
    loss = criterion(logits, y_train)  # Calculer la perte
    loss.backward()  # Passage arrière : calculer les gradients
    optimizer.step()  # Mettre à jour les poids du modèle

    scheduler.step(loss.item())  # Ajuster le taux d'apprentissage si nécessaire

    loss_values.append(loss.item())  # Enregistrer la perte pour la visualisation

    # Arrêt anticipé : vérifier si la perte actuelle est la meilleure
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Arrêt anticipé déclenché.")
            break

    # Afficher les progrès tous les 10 époques
    if (epoch + 1) % 10 == 0:
        print(f'Époque {epoch+1}, Perte : {loss.item()}')

# Évaluer le modèle sur le jeu de test
model.eval()  # Mettre le modèle en mode évaluation
with torch.no_grad():
    logits = model(X_test)  # Passage avant sur les données de test
    test_predictions = torch.round(torch.sigmoid(logits))  # Convertir les logits en prédictions binaires
    print("Prédictions sur le jeu de test :", test_predictions)

# Calculer les métriques de performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convertir les prédictions en tableaux numpy binaires
binary_predictions = test_predictions.numpy()

# Calculer les métriques
accuracy = accuracy_score(y_test.numpy(), binary_predictions)
precision = precision_score(y_test.numpy(), binary_predictions, average='macro', zero_division=0)
recall = recall_score(y_test.numpy(), binary_predictions, average='macro')
f1 = f1_score(y_test.numpy(), binary_predictions, average='macro')

print(f"Précision : {accuracy}")
print(f"Précision pondérée : {precision}")
print(f"Rappel : {recall}")
print(f"Score F1 : {f1}")

# Tracer la courbe de perte
import matplotlib.pyplot as plt

plt.plot(range(1, len(loss_values) + 1), loss_values)  # Tracer les valeurs de perte au fil des époques
plt.xlabel('Époque')  # Étiquette pour l'axe x
plt.ylabel('Perte')  # Étiquette pour l'axe y
plt.title('Courbe de Perte pendant l\'Entraînement')  # Titre du graphique
plt.show()
