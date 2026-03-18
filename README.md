**🛠️ ML Data Cleaning Skills — Mes fonctions personnelles
**
Ce que c'est : Mes outils réutilisables pour nettoyer des données et entraîner des modèles ML


🗂️ Structure du dépôt
ml-data-cleaning-skills/
│
├── 📄 README.md                  ← Ce fichier
├── 📁 my_functions/              ← 🌟 MES FONCTIONS PERSONNELLES
│   ├── data_cleaning.py          ← Nettoyage et exploration des données
│   ├── preprocessing.py          ← Standardisation, encodage, splits
│   └── ml_models.py              ← Modèles ML entraînables et évaluables
│
└── 📁 notebooks/                 ← Notebooks de référence (TP original, non personnel)
    └── PracticalWork_DataCleaning_clean.ipynb

📂 Contenu de my_functions/
🧹 data_cleaning.py — Nettoyage des données
FonctionCe qu'elle faitload_and_explore(filepath)Charge un CSV et affiche head, info, describe, nuniquedrop_irrelevant_columns(df, cols)Supprime les colonnes inutileshandle_missing_values(df, strategy)Gère les NaN : 'drop', 'fill', ou 'mean'bin_column(df, col, bins, labels)Transforme une colonne continue en catégoriesfix_category_typo(df, col, wrong, correct)Corrige une faute dans une catégorieadd_ratio_features(df)Crée des features ratios (feature engineering)drop_redundant_columns(df, threshold)Supprime les colonnes trop corrélées entre elles

⚙️ preprocessing.py — Préparation pour le ML
FonctionCe qu'elle faitsimple_split(df, test_size)Split aléatoire train/teststratified_split(df, col, test_size)Split stratifié (garde les proportions des classes)standardize_numerical(df, scaler)Standardise les features numériques (mean=0, std=1)one_hot_encode(df, categorical_cols)Encode les variables catégorielles en one-hotprocess_data(dataset, target, cat_cols)Pipeline complet : split X/Y + standardisation + one-hotto_tensorflow_dataset(X, Y, n_classes)Convertit en dataset TensorFlow prêt pour model.fit()

🤖 ml_models.py — Modèles Machine Learning
FonctionTypeCe qu'elle faittrain_linear_regression(...)RégressionBaseline linéaire, calcule le RMSEtrain_decision_tree_regressor(...)RégressionArbre de décision + feature importancestrain_svr(...)RégressionSupport Vector Machinetrain_decision_tree_classifier(...)ClassificationArbre + rapport de classificationtrain_random_forest(...)ClassificationEnsemble de plusieurs arbrestrain_mlp_sklearn(...)ClassificationRéseau de neurones scikit-learnbuild_tensorflow_model(...)ClassificationRéseau de neurones TensorFlow/Kerastrain_tensorflow_model(...)ClassificationEntraîne avec early stoppingplot_training_history(history)VisualisationCourbes loss train vs validation

📖 Concepts clés que j'ai appris
🔴 Data Leakage — le piège le plus important
Le data leakage = quand des informations du test set "fuient" dans l'entraînement.
Résultat : le modèle semble performant en évaluation mais échoue en production.
python# ❌ FAUX : le scaler voit les données de test → data leakage !
scaler.fit_transform(X_train)
scaler.fit_transform(X_test)  # Ne jamais re-fitter sur le test !

# ✅ CORRECT : fit seulement sur train, transform sur les deux
scaler.fit_transform(X_train)  # apprend la moyenne et std du train
scaler.transform(X_test)       # applique sans ré-apprendre

🔴 One-Hot Encoding vs Label Encoding
python# ❌ Label Encoding pour des catégories nominales
# "Aerial"=1, "Aquatic"=2, "Terrestrial"=3
# → le modèle pense que Terrestrial(3) > Aquatic(2) > Aerial(1)
# Ce n'est pas ce qu'on veut !

# ✅ One-Hot Encoding
# "Aerial"     → [1, 0, 0]
# "Aquatic"    → [0, 1, 0]
# "Terrestrial"→ [0, 0, 1]
# Aucune relation d'ordre → correct pour des catégories nominales

🔴 Split Stratifié vs Aléatoire
python# ❌ Split aléatoire avec des classes rares
# Si "Scavenger" = 0.3% des données et dataset = 10 000 lignes
# → seulement ~30 Scavengers. Le test set peut n'en avoir aucun !

# ✅ Split stratifié
# Garantit que chaque classe garde sa proportion dans train ET test
from sklearn.model_selection import StratifiedShuffleSplit

🔴 Overfitting — le détecter avec les courbes de loss
Loss
│  \
│   \  ← train loss descend  ✅
│    \_______
│            \____
│                 \   ← val loss remonte  ❌ = OVERFITTING
│
└────────────────── Epochs
La solution : Early Stopping → on arrête quand val_loss ne s'améliore plus.

🔴 Standardisation vs Normalisation
Standardisation (StandardScaler)Normalisation (MinMaxScaler)Résultatmean=0, std=1valeurs dans [0, 1]Avantageconserve la distributionfacile à interpréterInconvénientmoins intuitiftrès sensible aux outliersQuand l'utilisermodèles statistiques, SVM, MLPquand les outliers sont contrôlés

🚀 Comment utiliser ces fonctions
python# Import des modules
from my_functions.data_cleaning import load_and_explore, drop_irrelevant_columns, handle_missing_values
from my_functions.preprocessing import stratified_split, process_data
from my_functions.ml_models import train_random_forest, build_tensorflow_model

# 1. Charger et explorer
df = load_and_explore('AVONET.csv')

# 2. Nettoyer
df = drop_irrelevant_columns(df, ['Family3', 'Order3', 'Total.individuals'])
df = handle_missing_values(df, strategy='drop')

# 3. Splitter
train_set, test_set = stratified_split(df, stratify_col='Trophic.Niche', test_size=0.25)

# 4. Préparer les features
cat_cols = ['Habitat', 'Primary.Lifestyle', 'Centroid.Longitude']
X_train, Y_train, scaler = process_data(train_set, 'Trophic.Niche', cat_cols)
X_test,  Y_test,  _      = process_data(test_set,  'Trophic.Niche', cat_cols, scaler=scaler)

# 5. Entraîner un modèle
model, preds, report = train_random_forest(X_train, Y_train, X_test, Y_test)

📦 Dépendances
bashpip install pandas numpy matplotlib seaborn scikit-learn tensorflow

🔗 Liens utiles

📘 scikit-learn — Preprocessing
📘 scikit-learn — Model selection
📘 TensorFlow — Keras Sequential model
📘 Towards Data Science — Data Leakage
📗 Le TP original archivé
