# Détecteur de SPAM

Système local de classification automatique d'e-mails avec TF-IDF, Naive Bayes
et Mailpit. Le projet entraîne un modèle sur des messages étiquetés, puis
observe les e-mails reçus dans Mailpit afin de produire une décision, une
probabilité et des statistiques d'évaluation.

Projet universitaire réalisé dans le cadre d'un travail de recherche et de
mise en pratique de l'apprentissage automatique et du traitement du langage.

Date de présentation : **18 février 2026**

## Présentation

Le projet distingue deux classes :

- `ham` : message normal et légitime ;
- `spam` : message indésirable ou suspect.

La chaîne complète comprend la préparation des données, le nettoyage du texte,
la vectorisation TF-IDF, l'entraînement d'un classifieur Naive Bayes, puis
l'intégration avec Mailpit pour tester le modèle dans un environnement local.

Le système conserve aussi un historique des prédictions. Il est donc possible
d'analyser un message individuel ou un groupe de messages avec un `run_id`.

## Fonctionnalités

- fusion de plusieurs jeux de données de spam et de messages normaux ;
- nettoyage du contenu et normalisation linguistique ;
- représentation TF-IDF avec mots et groupes de mots ;
- comparaison de classifieurs Naive Bayes ;
- sélection d'un seuil de décision adapté à la détection du spam ;
- sauvegarde du modèle dans un fichier `joblib` ;
- réception de messages de test avec Mailpit ;
- ajout de tags de résultat et de confiance dans Mailpit ;
- évaluation individuelle ou par groupe de messages ;
- calcul de la matrice de confusion, de l'accuracy, de la précision, du rappel
  et du score F1 ;
- affichage des détails techniques : probabilité, seuil et mots influents.

## Stack technique

### Apprentissage automatique

- Python ;
- pandas et NumPy pour la manipulation des données ;
- scikit-learn pour TF-IDF, les séparations et les classifieurs ;
- NLTK pour la lemmatisation ;
- joblib pour sauvegarder et charger le modèle.

### Test et intégration

- Mailpit pour recevoir les messages localement ;
- SMTP pour envoyer les e-mails de test ;
- API HTTP de Mailpit pour lire les messages ;
- JSON et JSONL pour l'état, les métriques et l'historique.

## Structure du projet

```text
Détecteur de SPAM/
├── README.md
├── requirements.txt
├── Spam.py
├── over.txt              # source LaTeX du rapport universitaire
├── data/
│   ├── README.md         # provenance et préparation des datasets
│   ├── raw/              # jeux de données originaux pour l'entraînement
│   └── processed/        # données nettoyées exportées par Spam.py
├── models/
│   └── spam_tfidf_nb.joblib
└── Mailpit/
    ├── app.py            # surveillance, prédiction et tags
    ├── send.py           # envoi de messages de test
    ├── evaluate.py       # statistiques et rapports
    ├── data/             # fichiers de démonstration pour Mailpit
    ├── doc/              # rapport et documentation du projet
    ├── runtime/          # fichiers générés pendant l'exécution
    └── bin/              # exécutables Mailpit Linux et Windows
```

## Installation

### Prérequis

- Python 3 ;
- `pip` ;
- les dépendances listées dans `requirements.txt` ;
- Mailpit fourni dans `Mailpit/bin/`.

### 1. Installer les dépendances

Depuis la racine du projet :

```bash
python -m pip install -r requirements.txt
```

### 2. Vérifier les jeux de données

Les fichiers utilisés par le script d'entraînement sont :

```text
data/raw/spam_assassin.csv
data/raw/spam_ham_dataset.csv
```

Les colonnes sont normalisées par `Spam.py` avant la fusion des deux sources.
Les deux fichiers bruts sont versionnés dans le projet afin que les utilisateurs
puissent réentraîner le modèle après un clonage. Le fichier traité est généré
localement et reste exclu du dépôt. Voir [`data/README.md`](data/README.md)
pour les détails et la provenance des datasets.

## Entraînement du modèle

Depuis la racine du projet :

```bash
python Spam.py
```

Le script réalise les opérations suivantes :

1. chargement des deux jeux de données ;
2. harmonisation des colonnes et des étiquettes ;
3. nettoyage du contenu des e-mails ;
4. tokenisation et lemmatisation ;
5. séparation entraînement, validation et test ;
6. construction de la représentation TF-IDF ;
7. comparaison de plusieurs modèles et valeurs d'alpha ;
8. recherche du seuil de décision avec le score F2 ;
9. entraînement final et évaluation ;
10. sauvegarde de l'artefact du modèle.

Le modèle final est enregistré ici :

```text
models/spam_tfidf_nb.joblib
```

Cet artefact contient le vectoriseur, le classifieur, le seuil et les
informations nécessaires à l'application de détection.

## Lancer Mailpit

Mailpit est utilisé comme environnement local de réception. Les messages ne
sont pas envoyés vers de vrais destinataires.

### Linux

Depuis le dossier principal :

```bash
cd Mailpit
./bin/mailpit
```

### Windows

```powershell
cd Mailpit
.\bin\mailpit.exe
```

L'interface web est disponible sur :

```text
http://localhost:8025
```

Le serveur SMTP utilisé pour les tests écoute normalement sur le port `1025`.

## Démarrer le détecteur

Dans un autre terminal :

```bash
cd Mailpit
python app.py --reset-state
```

L'application surveille les nouveaux messages, charge le modèle depuis
`../models/spam_tfidf_nb.joblib`, puis ajoute des informations de résultat dans
Mailpit.

Les tags principaux sont :

- `resultat:normal` ou `resultat:spam` ;
- `confiance:faible`, `confiance:moyenne` ou `confiance:forte` ;
- `verification:correcte` ou `verification:incorrecte`.

## Données de démonstration

Les fichiers de test sont séparés des données utilisées pour entraîner le
modèle.

### Messages intégrés

```bash
cd Mailpit
python send.py templates --kind both --count 2 --interval 1
```

### Message personnalisé

```bash
python send.py message \
  --subject "Meeting tomorrow" \
  --body "Hello, the meeting starts at 9 AM." \
  --expected-label ham
```

### Envoi depuis un CSV

```bash
python send.py csv --csv-path data/demo_mails.csv --interval 1
```

Pour vérifier le contenu du fichier sans envoyer les messages :

```bash
python send.py csv --csv-path data/demo_mails.csv --dry-run
```

Le flux de démonstration documenté utilise `Mailpit/data/demo_mails.csv`.
L'autre fichier CSV présent dans ce dossier est conservé pour des essais
complémentaires et n'est pas nécessaire au fonctionnement principal.

## Statistiques et évaluation

### Analyser un groupe de messages

Le `run_id` identifie plusieurs messages appartenant au même scénario :

```bash
python evaluate.py --run-id RUN_ID
```

Pour obtenir les détails de chaque message :

```bash
python evaluate.py --run-id RUN_ID --details
```

### Analyser un message précis

```bash
python evaluate.py --message-id MESSAGE_ID
```

Le rapport affiche notamment :

- la classe attendue et la classe prédite ;
- la probabilité de spam ;
- le seuil utilisé ;
- le niveau de confiance ;
- les mots influents ;
- la matrice de confusion ;
- l'accuracy ;
- la précision, le rappel et le score F1 pour chaque classe.

## Fichiers générés

Le dossier `Mailpit/runtime/` contient les fichiers produits pendant
l'exécution :

- `state.json` : identifiants des messages déjà traités ;
- `metrics.json` : compteurs cumulés ;
- `predictions_log.jsonl` : détail technique des prédictions.

Ces fichiers sont des sorties d'exécution et ne remplacent pas les données
originales du dossier `data/raw/`.

## Documentation

La source du rapport universitaire est conservée dans `over.txt`. Le document
PDF final se trouve dans `Mailpit/doc/`.

## Tests et qualité

Vérifier la syntaxe Python depuis la racine :

```bash
python -m py_compile Spam.py Mailpit/app.py Mailpit/send.py Mailpit/evaluate.py
```

Un scénario complet peut être exécuté avec les commandes suivantes :

```bash
cd Mailpit
python send.py templates --kind both --count 2 --interval 1
python evaluate.py --run-id RUN_ID --details
```

Les scénarios recommandés sont :

- un message normal évident ;
- un spam évident ;
- un message normal ambigu ;
- un groupe contenant plusieurs messages normaux et spams.

## Sécurité et limites

- Mailpit doit rester limité à l'environnement local de test ;
- les vrais e-mails ne doivent pas être envoyés directement vers ce système
  sans validation ;
- les probabilités du modèle ne constituent pas une certitude ;
- un faux positif peut bloquer un message normal ;
- un faux négatif peut laisser passer un spam ;
- les messages utilisés pour l'entraînement doivent être contrôlés et
  dépourvus de données personnelles inutiles ;
- les données d'entraînement et le prétraitement doivent rester cohérents avec
  les messages reçus par l'application.

## Licence

Projet pédagogique et expérimental. La licence du projet sera définie
ultérieurement.
