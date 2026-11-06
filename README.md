# FarmView-IA

Application Streamlit de démonstration pour explorer un pipeline de diagnostic foliaire MobileNetV2. FarmView ne simule pas de diagnostic : en l'absence d'un modèle entraîné et compatible, la page Diagnostic indique que l'inférence est indisponible.

> **Limite actuelle :** aucun modèle entraîné, dataset validé, image de démonstration ou rapport de métriques n'est livré. PlantVillage comprend des photos de laboratoire ; une performance mesurée sur ce jeu ne garantit pas les performances sur des photos prises au champ. Les résultats nécessitent une confirmation par un agronome.

## État du projet

- Inférence prévue avec MobileNetV2 et vérification du fichier de classes.
- Préparation du dataset bloquée tant que la licence et les droits de réutilisation ne sont pas vérifiés manuellement.
- Aucun entraînement n'a été exécuté et aucune précision n'est revendiquée.
- Les données de la carte, de l'annuaire, des alertes et des graphiques du portail admin sont **des données de démonstration statiques**.
- Les exemples, métriques et résultats réels ne seront affichés que lorsque leurs artefacts source seront présents.

## Installation et lancement

Depuis la racine du dépôt `FarmView` (Windows PowerShell ou terminal) :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Le CI cible Python 3.11. Le paquet TensorFlow n'est pas déclaré pour Windows natif avec Python 3.11 ; pour l'entraînement, utilisez l'environnement Linux de Colab ou un environnement Linux compatible. Sans TensorFlow et sans modèle valide, l'application reste explicite et désactive tout diagnostic.

Le portail admin n'a pas d'identifiants par défaut. Configurez `FARMVIEW_ADMIN_EMAIL` et `FARMVIEW_ADMIN_PASSWORD`, ou les secrets Streamlit :

```toml
[admin]
email = "adresse-configuree"
password = "mot-de-passe-configure"
```

Ne commitez jamais ces secrets. Les valeurs d'environnement ont priorité sur les secrets Streamlit.

## Tests

Depuis la racine du dépôt :

```powershell
python -m pip install -r requirements-test.txt
python -m compileall -q app.py src scripts train_model.py tests
python -m pytest tests -q
```

Les tests du modèle réel, du rapport de métriques et de l'interface Streamlit sont ignorés lorsque leurs artefacts ou dépendances sont absents. Ces tests conditionnels ne prouvent pas la précision agronomique.

## Préparer un dataset et entraîner

La préparation nécessite une attestation explicite de la licence. Vérifiez la licence et les droits de réutilisation PlantVillage avant de passer ces valeurs ; elles constituent une déclaration de l'opérateur, pas un avis juridique.

Depuis la racine du dépôt :

```powershell
python scripts/prepare_dataset.py --license-confirmed --license-name "LICENCE VERIFIEE" --verification-evidence "URL et preuve vérifiée"
python train_model.py
```

Le dataset généré est ignoré par Git. La préparation produit un manifeste vérifié par l'opérateur et des partitions fixes train/val/test. Le modèle écrit `src/models/trained_models/farmview_model.keras` et les classes associées ; le rapport de jeu de test est écrit dans `reports/metrics.json` et `reports/confusion_matrix.png`. **Ces fichiers ne sont pas présents tant que le pipeline n'a pas été exécuté avec des données autorisées.**

`scripts/train_colab.py` est le point d'entrée prévu pour un notebook Colab. Ne publiez pas le modèle ou le dataset avant d'avoir confirmé leurs licences.

## Déploiement — checklist, après validation

1. Vérifier la licence de la source, le manifeste et l'absence de doublons entre classes.
2. Exécuter l'entraînement avec les partitions fixes ; réserver la partition test à l'évaluation finale.
3. Examiner `reports/metrics.json` et la matrice de confusion. Ne publier que les mesures réellement obtenues et leur protocole.
4. Tester avec le modèle exporté et des images retenues indépendamment ; documenter les faux positifs et faux négatifs.
5. Vérifier la taille de `farmview_model.keras` ; utiliser Git LFS si le fichier dépasse la limite GitHub applicable.
6. Configurer Python 3.11, `app.py` et les secrets admin dans Streamlit Community Cloud.
7. Après accord explicite sur le design et les tests, effectuer un test de fumée avec les exemples autorisés.
8. Ne pas déployer avant validation de l'utilisateur.

## Fiche portfolio (état honnête)

- **Titre :** FarmView-IA
- **Accroche :** Pipeline de diagnostic foliaire qui distingue explicitement une prédiction réelle d'un modèle indisponible.
- **Description :** FarmView-IA est une application Streamlit consacrée à l'exploration du diagnostic foliaire. Elle intègre un pipeline MobileNetV2 prévu pour vérifier la compatibilité des classes et refuser les résultats de substitution lorsque le modèle est absent ou invalide. La préparation des données et l'évaluation sur un jeu de test réservé sont encadrées par des garde-fous de licence et de traçabilité. Aucun modèle réel ni score de performance n'étant disponible à ce stade, aucune précision agronomique n'est revendiquée.
- **Stack :** Python, Streamlit, TensorFlow/Keras, MobileNetV2, Pillow, NumPy, OpenCV, pytest
- **Rôle :** Conception de l'application, du pipeline de préparation/entraînement et des tests.
- **Liens :** [GitHub](https://github.com/oumar2707/FarmView) ; démo en ligne : non déployée.
- **Captures/GIF :** à produire après validation visuelle ; aucune capture de démonstration n'est inventée.

```json
{
  "title": "FarmView-IA",
  "tagline": "Pipeline de diagnostic foliaire qui distingue une prédiction réelle d'un modèle indisponible.",
  "description": "Application Streamlit de diagnostic foliaire intégrant un pipeline MobileNetV2 avec vérification des classes, préparation des données et évaluation sur un jeu de test réservé. Aucun modèle réel ni score de performance n'est disponible pour le moment ; aucune précision agronomique n'est revendiquée.",
  "stack": ["Python", "Streamlit", "TensorFlow/Keras", "MobileNetV2", "Pillow", "NumPy", "OpenCV", "pytest"],
  "demoUrl": null,
  "repoUrl": "https://github.com/oumar2707/FarmView",
  "images": [],
  "metrics": {}
}
```

## Licence

Le code source est distribué sous licence MIT (voir [LICENSE](./LICENSE)). Cela ne confère aucun droit sur le dataset PlantVillage, qui doit être vérifié séparément.
