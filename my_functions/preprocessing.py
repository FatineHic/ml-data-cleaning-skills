"""
================================================================
preprocessing.py — Mes fonctions de préparation des données ML
================================================================
Dataset utilisé : AVONET (mesures morphologiques d'espèces d'oiseaux)

Ces fonctions transforment les données brutes en données
prêtes à être ingérées par un modèle Machine Learning.

Langage   : Python 3
Librairies: pandas, numpy, scikit-learn, tensorflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf


# ─────────────────────────────────────────────
# 1. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────

def simple_split(df, test_size=0.25, seed=42):
    """
    Divise le dataset en ensemble d'entraînement et de test (split aléatoire).

    Pourquoi : on ne doit JAMAIS évaluer un modèle sur les données
    qui ont servi à l'entraîner. Sinon on mesure la mémorisation,
    pas la généralisation (= overfitting).

    ⚠️ Problème : si les classes sont déséquilibrées, le split aléatoire
    peut créer un test set sans certaines classes rares. Utiliser
    stratified_split() dans ce cas.

    Args:
        df (pd.DataFrame): le dataframe complet
        test_size (float): proportion de données de test (ex: 0.25 = 25%)
        seed (int): graine pour la reproductibilité

    Returns:
        tuple: (train_set, test_set) deux DataFrames
    """
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=seed)
    print(f"✅ Split aléatoire : {len(train_set)} train / {len(test_set)} test")
    return train_set, test_set


def stratified_split(df, stratify_col, test_size=0.25, seed=42):
    """
    Divise le dataset en gardant les proportions de chaque classe.

    Pourquoi : si une classe a très peu d'exemples (ex: 2% des données),
    un split aléatoire pourrait la "rater" dans le test set.
    Le split stratifié GARANTIT que chaque classe est représentée
    proportionnellement dans train ET test.

    Utilise StratifiedShuffleSplit de scikit-learn.

    Args:
        df (pd.DataFrame): le dataframe complet
        stratify_col (str): colonne sur laquelle stratifier (souvent la cible)
        test_size (float): proportion de données de test
        seed (int): graine pour la reproductibilité

    Returns:
        tuple: (train_set, test_set) deux DataFrames
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_idx, test_idx in split.split(df, df[stratify_col]):
        train_set = df.loc[train_idx]
        test_set  = df.loc[test_idx]

    print(f"✅ Split stratifié sur '{stratify_col}' :")
    print(f"   {len(train_set)} train / {len(test_set)} test")
    return train_set, test_set


# ─────────────────────────────────────────────
# 2. STANDARDISATION DES FEATURES NUMÉRIQUES
# ─────────────────────────────────────────────

def standardize_numerical(df, scaler=None):
    """
    Standardise les colonnes numériques (mean=0, std=1).

    Pourquoi : les modèles ML sont sensibles à l'échelle des données.
    Une feature qui varie de 0 à 100000 "écrase" une feature qui varie
    de 0 à 1 si on ne les normalise pas. Deux approches :

    - Scaling (MinMaxScaler)   : met tout entre [0, 1] → sensible aux outliers
    - Standardisation (StandardScaler) : mean=0, std=1 → conserve la distribution

    ⚠️ IMPORTANT : le scaler doit être FIT uniquement sur le train set,
    puis APPLIQUÉ (transform seulement) sur le test set.
    Sinon = data leakage (fuite d'info du test vers le train).

    Args:
        df (pd.DataFrame): le dataframe (uniquement les colonnes numériques)
        scaler: un StandardScaler déjà fitté (None = en créer un nouveau)

    Returns:
        tuple: (df_standardisé, scaler) pour pouvoir ré-utiliser le scaler
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    X_np = df[numeric_cols].values

    if scaler is None:
        # Fit + transform (sur le train set)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_np)
        print("✅ Scaler fitté et appliqué (train set)")
    else:
        # Transform seulement (sur le test set) → pas de leakage !
        X_std = scaler.transform(X_np)
        print("✅ Scaler existant appliqué (test set — pas de re-fit)")

    df_std = pd.DataFrame(X_std, columns=numeric_cols, index=df.index)
    return df_std, scaler


# ─────────────────────────────────────────────
# 3. ENCODAGE ONE-HOT DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────

def one_hot_encode(df, categorical_cols):
    """
    Encode les variables catégorielles en vecteurs one-hot.

    Pourquoi : un modèle ML ne peut pas traiter du texte (ex: "Carnivore").
    Option 1 — Label encoding : Carnivore=1, Herbivore=2, Omnivore=3
        ❌ Problème : le modèle pense que Omnivore(3) > Herbivore(2) > Carnivore(1)
           ce qui n'a aucun sens pour des catégories nominales.
    Option 2 — One-hot encoding : chaque catégorie → une colonne binaire (0 ou 1)
        ✅ Correct : Carnivore = [1, 0, 0], Herbivore = [0, 1, 0], etc.

    pd.get_dummies() fait ça automatiquement.

    Args:
        df (pd.DataFrame): le dataframe
        categorical_cols (list): liste des colonnes à encoder

    Returns:
        tuple: (df_sans_catégories, dict_des_one_hots)
               Le dict permet de concaténer facilement après
    """
    one_hots = {}
    df = df.copy()

    for col in categorical_cols:
        one_hots[col] = pd.get_dummies(df[col], prefix=col)
        df.drop(columns=[col], inplace=True)
        print(f"✅ One-hot encodé : '{col}' → {list(one_hots[col].columns)}")

    return df, one_hots


# ─────────────────────────────────────────────
# 4. PIPELINE COMPLET DE PRÉPARATION
# ─────────────────────────────────────────────

def process_data(dataset, target_col, categorical_cols, scaler=None):
    """
    Pipeline complet de préparation des données pour le ML.

    Effectue dans l'ordre :
      1. Séparation X (features) / Y (cible)
      2. Standardisation des colonnes numériques
      3. Encodage one-hot des colonnes catégorielles
      4. Suppression des colonnes catégorielles originales
      5. Concaténation : données standardisées + one-hots

    ⚠️ À appeler séparément sur train et test (en passant le scaler
    du train au test pour éviter le data leakage).

    Exemple d'utilisation :
        X_train, Y_train, scaler = process_data(train_set, 'Trophic.Niche', cat_cols)
        X_test,  Y_test,  _      = process_data(test_set,  'Trophic.Niche', cat_cols, scaler=scaler)

    Args:
        dataset (pd.DataFrame): le dataset (train ou test)
        target_col (str): nom de la colonne cible
        categorical_cols (list): colonnes catégorielles à encoder
        scaler: scaler déjà fitté sur train (None = créer un nouveau)

    Returns:
        tuple: (X, Y, scaler)
    """
    dataset = dataset.copy()

    # 1. Séparer X et Y
    Y = dataset[target_col].copy()
    X = dataset.drop(columns=[target_col])

    # 2. Standardiser les numériques
    X_num = X.select_dtypes(include=np.number)
    X_std, scaler = standardize_numerical(X_num, scaler=scaler)

    # 3. One-hot encoder les catégorielles
    one_hots = {}
    for col in categorical_cols:
        if col in X.columns:
            one_hots[col] = pd.get_dummies(X[col], prefix=col)

    # 4. Concaténer tout
    parts = [X_std] + list(one_hots.values())
    X_final = pd.concat(parts, axis=1)

    print(f"✅ Dataset préparé : {X_final.shape[0]} exemples × {X_final.shape[1]} features")
    return X_final, Y, scaler


# ─────────────────────────────────────────────
# 5. CONVERSION EN DATASET TENSORFLOW
# ─────────────────────────────────────────────

def to_tensorflow_dataset(X, Y, n_classes, batch_size=32, shuffle=True):
    """
    Convertit des DataFrames pandas en dataset TensorFlow.

    Pourquoi : TensorFlow ne consomme pas directement des DataFrames pandas.
    On doit convertir en tf.data.Dataset, ce qui permet aussi de :
    - mélanger les données (shuffle)
    - les regrouper en batchs
    - les traiter en pipeline efficacement

    L'encodage one-hot de Y est nécessaire pour la classification multi-classes
    avec CategoricalCrossentropy. LabelEncoder transforme les textes en entiers,
    puis to_categorical crée le vecteur one-hot.

    Args:
        X (pd.DataFrame): features d'entrée
        Y (pd.Series): cibles (texte ou numérique)
        n_classes (int): nombre de classes (pour le one-hot de Y)
        batch_size (int): taille des batchs
        shuffle (bool): mélanger les données ou non

    Returns:
        tf.data.Dataset: dataset prêt pour model.fit()
    """
    # Encode les labels texte → entiers → one-hot
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y)
    Y_onehot  = tf.keras.utils.to_categorical(Y_encoded, n_classes)

    # Créer le dataset TensorFlow
    X_array = np.asarray(X).astype('float32')
    ds = tf.data.Dataset.from_tensor_slices((X_array, Y_onehot))

    if shuffle:
        ds = ds.shuffle(len(X))

    ds = ds.batch(batch_size)
    print(f"✅ Dataset TensorFlow créé : {len(X)} exemples, batchs de {batch_size}")
    return ds, encoder
