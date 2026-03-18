"""
========================================================
data_cleaning.py — Mes fonctions de nettoyage de données
========================================================
Dataset utilisé : AVONET (mesures morphologiques d'espèces d'oiseaux)

Langage   : Python 3
Librairies: pandas, numpy
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# 1. CHARGEMENT & EXPLORATION RAPIDE
# ─────────────────────────────────────────────

def load_and_explore(filepath, seed=42):
    """
    Charge un fichier CSV et affiche les infos de base.

    Pourquoi : avant tout traitement, il faut comprendre ce qu'on a.
    - head()     → voir les premières lignes
    - info()     → types, valeurs manquantes
    - describe() → statistiques de base (min, max, moyenne...)
    - nunique()  → nombre de valeurs uniques par colonne

    Args:
        filepath (str): chemin vers le fichier CSV
        seed (int): graine aléatoire pour la reproductibilité

    Returns:
        pd.DataFrame: le dataframe chargé
    """
    np.random.seed(seed)
    df = pd.read_csv(filepath)

    print("=== APERÇU DES DONNÉES ===")
    print(f"Taille : {df.shape[0]} lignes × {df.shape[1]} colonnes\n")
    print("--- Premières lignes ---")
    print(df.head())
    print("\n--- Types et valeurs manquantes ---")
    print(df.info())
    print("\n--- Statistiques ---")
    print(df.describe())
    print("\n--- Valeurs uniques par colonne ---")
    print(df.nunique())

    return df


# ─────────────────────────────────────────────
# 2. SUPPRESSION DES COLONNES INUTILES
# ─────────────────────────────────────────────

def drop_irrelevant_columns(df, columns_to_drop):
    """
    Supprime les colonnes qui ne sont pas utiles pour l'analyse.

    Pourquoi : réduire la dimensionnalité et éviter le bruit.
    Exemples de colonnes à supprimer : identifiants, métadonnées,
    colonnes avec trop de valeurs manquantes, colonnes redondantes.

    Args:
        df (pd.DataFrame): le dataframe original
        columns_to_drop (list): liste des noms de colonnes à supprimer

    Returns:
        pd.DataFrame: dataframe sans les colonnes supprimées
    """
    df = df.drop(columns=columns_to_drop)
    print(f"✅ Colonnes supprimées : {columns_to_drop}")
    print(f"   Taille restante : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────
# 3. GESTION DES VALEURS MANQUANTES (NaN)
# ─────────────────────────────────────────────

def handle_missing_values(df, strategy='drop', fill_value=None):
    """
    Gère les valeurs manquantes (NaN) dans le dataframe.

    Pourquoi : les modèles ML ne peuvent PAS traiter les NaN.
    On a 3 options :
      - 'drop'  → supprime les lignes avec NaN (simple, mais perd des données)
      - 'fill'  → remplace les NaN par une valeur (fill_value)
      - 'mean'  → remplace les NaN par la moyenne de chaque colonne numérique

    Args:
        df (pd.DataFrame): le dataframe
        strategy (str): 'drop', 'fill', ou 'mean'
        fill_value: valeur de remplacement si strategy='fill'

    Returns:
        pd.DataFrame: dataframe sans valeurs manquantes
    """
    initial_rows = len(df)

    if strategy == 'drop':
        df = df.dropna()
        print(f"✅ Lignes supprimées (NaN) : {initial_rows - len(df)}")

    elif strategy == 'fill':
        df = df.fillna(fill_value)
        print(f"✅ NaN remplacés par : {fill_value}")

    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print(f"✅ NaN numériques remplacés par la moyenne")

    print(f"   Taille finale : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    return df


# ─────────────────────────────────────────────
# 4. TRANSFORMATION D'UNE COLONNE CONTINUE EN CATÉGORIES
# ─────────────────────────────────────────────

def bin_column(df, column, bins, labels):
    """
    Transforme une colonne numérique continue en catégories (bins).

    Pourquoi : certaines variables continues sont mieux interprétées
    comme des catégories (ex: longitude → Ouest / Milieu / Est).
    pd.cut() divise les valeurs en intervalles définis par 'bins'.

    Exemple :
        bin_column(df, 'Centroid.Longitude',
                   bins=[-200, -25, 60, np.inf],
                   labels=['West', 'Greenwich', 'East'])

    Args:
        df (pd.DataFrame): le dataframe
        column (str): nom de la colonne à transformer
        bins (list): les bords des intervalles (ex: [-200, -25, 60, inf])
        labels (list): les noms des catégories créées

    Returns:
        pd.DataFrame: dataframe avec la colonne transformée
    """
    df[column] = pd.cut(df[column], bins=bins, labels=labels)
    print(f"✅ Colonne '{column}' transformée en catégories : {labels}")
    return df


# ─────────────────────────────────────────────
# 5. CORRECTION D'UNE VALEUR CATÉGORIELLE MAL SAISIE
# ─────────────────────────────────────────────

def fix_category_typo(df, column, wrong_value, correct_value):
    """
    Corrige une faute de frappe ou une incohérence dans une colonne catégorielle.

    Pourquoi : 'Shrubland ' (avec espace) et 'Shrubland' (sans espace)
    seraient traitées comme 2 catégories différentes par le modèle !
    Il faut nettoyer ça avant l'encodage.

    Args:
        df (pd.DataFrame): le dataframe
        column (str): colonne à corriger
        wrong_value: valeur incorrecte à remplacer
        correct_value: valeur correcte

    Returns:
        pd.DataFrame: dataframe corrigé
    """
    df[column] = df[column].replace(wrong_value, correct_value)
    print(f"✅ '{wrong_value}' → '{correct_value}' dans la colonne '{column}'")
    return df


# ─────────────────────────────────────────────
# 6. CRÉATION DE NOUVELLES FEATURES (FEATURE ENGINEERING)
# ─────────────────────────────────────────────

def add_ratio_features(df):
    """
    Ajoute des features calculées comme ratios entre colonnes existantes.

    Pourquoi : parfois le RATIO entre deux mesures est plus informatif
    que chaque mesure seule. Ex: la forme d'un bec (largeur/longueur)
    est plus révélatrice que sa largeur ou longueur seule.

    Features créées (adaptées au dataset AVONET) :
      - kipps_on_winglength   : Kipps.Distance / Wing.Length
      - Beak.nares_to_culmen  : Beak.Length_Nares / Beak.Length_Culmen
      - Beak.width_to_lculmen : Beak.Width / Beak.Length_Culmen
      - Beak.depth_to_lculmen : Beak.Depth / Beak.Length_Culmen

    Args:
        df (pd.DataFrame): le dataframe

    Returns:
        pd.DataFrame: dataframe avec les nouvelles colonnes
    """
    df['kipps_on_winglength']   = df['Kipps.Distance'] / df['Wing.Length']
    df['Beak.nares_to_culmen']  = df['Beak.Length_Nares'] / df['Beak.Length_Culmen']
    df['Beak.width_to_lculmen'] = df['Beak.Width'] / df['Beak.Length_Culmen']
    df['Beak.depth_to_lculmen'] = df['Beak.Depth'] / df['Beak.Length_Culmen']
    print("✅ 4 nouvelles features ratio ajoutées")
    return df


# ─────────────────────────────────────────────
# 7. SUPPRESSION DES COLONNES REDONDANTES
# ─────────────────────────────────────────────

def drop_redundant_columns(df, corr_threshold=0.95, target_col=None):
    """
    Supprime les colonnes numériques fortement corrélées entre elles.

    Pourquoi : deux colonnes très corrélées (ex: r > 0.95) apportent
    essentiellement la même information. Les garder toutes les deux :
      1. augmente inutilement le nombre de paramètres
      2. peut causer des problèmes de multicolinéarité
      3. rend le modèle moins interprétable

    Cette fonction cherche les paires de colonnes avec |corrélation| > seuil
    et supprime une des deux (celle qui est le moins corrélée à la cible).

    Args:
        df (pd.DataFrame): le dataframe
        corr_threshold (float): seuil de corrélation (ex: 0.95)
        target_col (str): colonne cible à ne pas supprimer

    Returns:
        pd.DataFrame: dataframe avec colonnes redondantes supprimées
        list: liste des colonnes supprimées
    """
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().abs()

    # Matrice triangulaire supérieure pour éviter les doublons
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Colonnes à supprimer (fortement corrélées à une autre)
    to_drop = [
        col for col in upper.columns
        if any(upper[col] > corr_threshold) and col != target_col
    ]

    df = df.drop(columns=to_drop)
    print(f"✅ Colonnes redondantes supprimées (seuil={corr_threshold}) : {to_drop}")
    return df, to_drop
