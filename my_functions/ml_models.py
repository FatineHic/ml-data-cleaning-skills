"""
=======================================================
ml_models.py — Mes fonctions d'entraînement de modèles
=======================================================
Dataset utilisé : AVONET (mesures morphologiques d'espèces d'oiseaux)

Ce fichier contient les modèles que j'ai compris et que je sais utiliser :
- Régression linéaire
- Arbre de décision (régression et classification)
- SVM
- Random Forest
- Réseau de neurones (scikit-learn MLP et TensorFlow)

Langage   : Python 3
Librairies: scikit-learn, tensorflow, numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (mean_squared_error, classification_report,
                             confusion_matrix, accuracy_score)
import tensorflow as tf


# ─────────────────────────────────────────────
# MODÈLES DE RÉGRESSION (prédire une valeur continue)
# ─────────────────────────────────────────────

def train_linear_regression(X_train, Y_train, X_test, Y_test):
    """
    Régression linéaire — le modèle le plus simple.

    Pourquoi commencer par ça : c'est la baseline. Si un modèle complexe
    ne fait pas mieux que la régression linéaire, c'est suspect.

    Principe : trouve la droite (ou hyperplan) qui minimise la somme des
    erreurs au carré entre les prédictions et les vraies valeurs.

    Métrique : RMSE (Root Mean Squared Error) — erreur moyenne en unités
    de la variable cible. Plus c'est bas, mieux c'est.

    Args:
        X_train, Y_train: données d'entraînement
        X_test, Y_test: données de test

    Returns:
        tuple: (modèle entraîné, RMSE)
    """
    model = LinearRegression()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))

    print(f"📊 Régression Linéaire — RMSE : {rmse:.2f}")
    return model, rmse


def train_decision_tree_regressor(X_train, Y_train, X_test, Y_test, max_depth=None):
    """
    Arbre de décision pour la régression.

    Pourquoi : peut capturer des relations non-linéaires (contrairement
    à la régression linéaire). Facile à interpréter avec feature_importances_.

    ⚠️ Attention : sans max_depth, l'arbre peut mémoriser tout le train set
    (overfitting) → il fait 0 d'erreur sur train mais mal sur test !

    Feature importance : chaque feature reçoit un score entre 0 et 1,
    qui représente sa contribution à la prédiction. Utile pour comprendre
    quelles variables comptent vraiment.

    Args:
        X_train, Y_train: données d'entraînement
        X_test, Y_test: données de test
        max_depth (int): profondeur max de l'arbre (None = illimité = risque d'overfitting)

    Returns:
        tuple: (modèle entraîné, RMSE)
    """
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))

    print(f"📊 Decision Tree Régression — RMSE : {rmse:.2f}")

    # Afficher les features les plus importantes
    importance = sorted(
        zip(model.feature_importances_, X_test.columns),
        reverse=True
    )
    print("\n🔍 Top 5 features importantes :")
    for score, name in importance[:5]:
        print(f"   {name:30} {score:.3f}")

    return model, rmse


def train_svr(X_train, Y_train, X_test, Y_test):
    """
    Support Vector Machine pour la régression (SVR).

    Pourquoi : peut trouver des frontières de décision complexes via
    le "kernel trick" (transformation dans un espace de plus haute dimension).

    ⚠️ Points importants :
    - Très lent sur de gros datasets
    - Très sensible à l'échelle → TOUJOURS standardiser avant !
    - gamma='auto' = 1 / n_features

    Args:
        X_train, Y_train: données d'entraînement (standardisées !)
        X_test, Y_test: données de test (standardisées !)

    Returns:
        tuple: (modèle entraîné, RMSE)
    """
    model = SVR(gamma='auto')
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(Y_test, predictions))

    print(f"📊 SVR — RMSE : {rmse:.2f}")
    return model, rmse


# ─────────────────────────────────────────────
# MODÈLES DE CLASSIFICATION (prédire une catégorie)
# ─────────────────────────────────────────────

def train_decision_tree_classifier(X_train, Y_train, X_test, Y_test, max_depth=4):
    """
    Arbre de décision pour la classification.

    Même principe que pour la régression mais la feuille finale prédit
    une classe (la majorité) plutôt qu'une valeur continue.

    Évaluation :
    - Accuracy : % de bonnes prédictions globalement
    - Classification report : precision, recall, f1 par classe
    - Confusion matrix : tableau NxN montrant les prédictions vs réalité
        - Ligne = classe réelle, Colonne = classe prédite
        - Diagonale = bonnes prédictions

    Args:
        X_train, Y_train: données d'entraînement
        X_test, Y_test: données de test
        max_depth (int): profondeur max de l'arbre

    Returns:
        tuple: (modèle, predictions, rapport)
    """
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_pred, Y_test)

    print(f"📊 Decision Tree Classifieur (max_depth={max_depth})")
    print(report)

    return model, y_pred, report


def train_random_forest(X_train, Y_train, X_test, Y_test, max_depth=10, n_estimators=100):
    """
    Random Forest pour la classification.

    Pourquoi c'est mieux qu'un seul arbre :
    Le Random Forest entraîne PLUSIEURS arbres (n_estimators) sur des
    sous-ensembles aléatoires des données, puis fait un vote majoritaire.
    Cela réduit l'overfitting et améliore la généralisation.

    C'est un exemple d'**ensemble method** : combiner plusieurs modèles
    faibles pour créer un modèle fort.

    Args:
        X_train, Y_train: données d'entraînement
        X_test, Y_test: données de test
        max_depth (int): profondeur max de chaque arbre
        n_estimators (int): nombre d'arbres dans la forêt

    Returns:
        tuple: (modèle, predictions, rapport)
    """
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_pred, Y_test)

    print(f"📊 Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
    print(report)

    return model, y_pred, report


def train_mlp_sklearn(X_train, Y_train, X_test, Y_test,
                      hidden_layers=(55, 5), max_iter=100):
    """
    Multi-Layer Perceptron (réseau de neurones) de scikit-learn.

    Pourquoi scikit-learn et pas TensorFlow ? Pour des réseaux simples,
    sklearn MLP est plus rapide à coder. TensorFlow est nécessaire pour
    des architectures complexes (CNN, RNN, Transfer Learning...).

    Hyperparamètres importants :
    - hidden_layer_sizes : architecture du réseau, ex (55, 5) = 2 couches cachées
    - solver : 'lbfgs' pour petits datasets, 'adam' pour grands
    - alpha : terme de régularisation L2 (évite l'overfitting)
    - max_iter : nombre d'itérations max

    Args:
        X_train, Y_train: données d'entraînement
        X_test, Y_test: données de test
        hidden_layers (tuple): tailles des couches cachées
        max_iter (int): iterations max

    Returns:
        tuple: (modèle, predictions, rapport)
    """
    model = MLPClassifier(
        solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=hidden_layers,
        random_state=1,
        max_iter=max_iter
    )
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_pred, Y_test)

    print(f"📊 MLP sklearn (couches={hidden_layers})")
    print(report)

    return model, y_pred, report


# ─────────────────────────────────────────────
# RÉSEAU DE NEURONES TENSORFLOW
# ─────────────────────────────────────────────

def build_tensorflow_model(input_dim, n_classes, hidden_units=32, dropout=0.3):
    """
    Construit un réseau de neurones dense avec TensorFlow/Keras.

    Architecture :
        Input → Dense(hidden_units, relu) → Dense(hidden_units, relu)
               → Dropout → Dense(n_classes, softmax)

    Couches importantes :
    - Dense(n, relu) : couche entièrement connectée avec activation ReLU
        relu(x) = max(0, x) — élimine les valeurs négatives, rapide à calculer
    - Dropout(rate) : désactive aléatoirement 'rate'% des neurones à chaque étape
        → force le réseau à ne pas trop dépendre de neurones spécifiques (régularisation)
    - Dense(n_classes, softmax) : couche de sortie pour classification multi-classes
        softmax convertit les logits en probabilités (somme = 1)

    Fonctions de perte pour la classification multi-classes :
    - categorical_crossentropy       : standard, supposé classes équilibrées
    - categorical_focal_crossentropy : donne plus de poids aux exemples difficiles
                                       (bien quand les classes sont déséquilibrées)

    Args:
        input_dim (int): nombre de features en entrée
        n_classes (int): nombre de classes à prédire
        hidden_units (int): nombre de neurones dans les couches cachées
        dropout (float): taux de dropout (0 à 1)

    Returns:
        tf.keras.Model: modèle compilé, prêt pour .fit()
    """
    f1_macro = tf.metrics.F1Score(average='macro', name='macro_F1')

    def top_2_accuracy(y_true, y_pred):
        return tf.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_units, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_units, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_focal_crossentropy',
        metrics=['accuracy', f1_macro, top_2_accuracy]
    )

    print(f"✅ Modèle TensorFlow construit")
    model.summary()
    return model


def train_tensorflow_model(model, train_ds, val_ds, epochs=100, patience=20):
    """
    Entraîne un modèle TensorFlow avec Early Stopping.

    Early Stopping : arrête l'entraînement quand la validation loss
    ne s'améliore plus pendant 'patience' epochs.

    Pourquoi c'est important :
    - Sans early stopping → le modèle continue à "mémoriser" le train set
      même quand il commence à mal généraliser (overfitting)
    - La validation loss augmente → signal d'overfitting → on arrête

    On surveille 'val_loss' et non 'loss' car c'est sur les données de
    validation qu'on mesure la vraie capacité de généralisation.

    Args:
        model: modèle compilé (depuis build_tensorflow_model)
        train_ds: dataset tf d'entraînement
        val_ds: dataset tf de validation
        epochs (int): nombre max d'epochs
        patience (int): epochs à attendre sans amélioration avant d'arrêter

    Returns:
        history: historique d'entraînement (loss, accuracy par epoch)
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=patience
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[early_stop]
    )

    return history


def plot_training_history(history):
    """
    Affiche les courbes de loss (train vs validation) après l'entraînement.

    Pourquoi c'est utile :
    - Si train_loss descend mais val_loss remonte → OVERFITTING
    - Si les deux descendent ensemble → bonne généralisation
    - Si aucune ne descend → underfitting (modèle trop simple ou mauvais LR)

    Args:
        history: objet retourné par model.fit()
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'],     label='Train loss',      color='blue')
    plt.plot(history.history['val_loss'], label='Validation loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbe de loss — Train vs Validation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
