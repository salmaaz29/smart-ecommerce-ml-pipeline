# 🛍️ Smart eCommerce Intelligence
### ML & DM Pipelines, A2A Agents, and LLMs

> Projet pédagogique — FST Tanger · Filière LSI 2 · Module DM & SID · 2025/2026

---

## 📋 Description

Système intelligent et automatisé capable de :
- **Scraper** des données produits sur des sites Shopify et WooCommerce
- **Analyser** les produits et identifier les meilleurs (Top-K)
- **Orchestrer** un pipeline ML avec Kubeflow sur Kubernetes
- **Visualiser** les résultats dans un dashboard de Business Intelligence
- **Enrichir** l'analyse avec des LLMs (LLaMA 3.3 via Groq)
- **Réfléchir** à l'architecture responsable selon le Model Context Protocol (Anthropic)

---

## 🏗️ Architecture du projet

```
smart_ecommerce/
│
├── scrapers/
│   ├── scraper_kiabi.py          # Scraping Kiabi (Shopify)
│   ├── scraper_beautymarket.py   # Scraping BeautyMarket (Shopify)
│   ├── scraper_justyol.py        # Scraping Justyol (Shopify)
│   └── scraper_woocommerce.py    # Scraping Lasolda + Lhmiza (WooCommerce)
│
├── data/
│   ├── dataset_kiabi_final.csv
│   ├── dataset_beautymarket.csv
│   ├── dataset_justyol.csv
│   ├── dataset_woocommerce.csv
│   ├── dataset_global.csv
│   ├── dataset_clean.csv
│   └── dataset_features.csv
│
├── prepa_data.py                 # Fusion + nettoyage + feature engineering
├── ml_pipeline_complet.py        # KMeans + Random Forest + Apriori + DBSCAN + PCA
├── kubeflow_pipeline.py          # Pipeline Kubeflow KFP complet
├── smart_ecommerce_pipeline.yaml # Pipeline compilé pour Kubeflow
├── dashboard.py                  # Dashboard Streamlit 6 onglets
│
└── outputs/
    ├── kubeflow_export.csv
    ├── kubeflow_topk.csv
    ├── kubeflow_anomalies.csv
    └── kubeflow_rapport.json
```

---

## 🔧 Technologies utilisées

| Catégorie | Outils |
|---|---|
| **Scraping** | Selenium, BeautifulSoup, Requests |
| **Data Processing** | Pandas, NumPy |
| **ML / DM** | Scikit-learn, MLxtend |
| **Algorithmes** | KMeans, Random Forest, Apriori, DBSCAN, PCA |
| **Orchestration ML** | Kubeflow Pipelines (KFP SDK 2.7.0) |
| **Infrastructure** | Kubernetes, Minikube, Docker |
| **Dashboard** | Streamlit, Plotly |
| **LLM** | LLaMA 3.3 70B via Groq API |
| **CI/CD** | GitHub Actions |

---

## 📊 Résultats ML

| Modèle | Métrique | Résultat |
|---|---|---|
| KMeans K=5 | Silhouette Score | 0.349 |
| Random Forest | Accuracy | 95% |
| Random Forest | F1 Score (CV 5-fold) | 0.704 |
| Apriori | Règles trouvées | 2520 |
| DBSCAN | Anomalies détectées | 8 (0.1%) |
| PCA | Variance expliquée | 60% |

---

## 🚀 Installation et exécution

### 1. Cloner le projet

```bash
git clone https://github.com/ton-username/smart-ecommerce-intelligence.git
cd smart-ecommerce-intelligence
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Scraping des données

```bash
python scrapers/scraper_kiabi.py
python scrapers/scraper_beautymarket.py
python scrapers/scraper_justyol.py
python scrapers/scraper_woocommerce.py
```

### 4. Nettoyage et feature engineering

```bash
python prepa_data.py
```

### 5. Pipeline ML complet

```bash
python ml_pipeline_complet.py
```

### 6. Pipeline Kubeflow

```bash
# Installation dépendances
pip install kfp==2.7.0

# Exécution locale
python kubeflow_pipeline.py --local

# Compilation en YAML
python kubeflow_pipeline.py --compile

# Upload sur Kubeflow (nécessite Minikube + Kubeflow installés)
python kubeflow_pipeline.py --upload
```

### 7. Dashboard Streamlit

```bash
pip install streamlit plotly groq
streamlit run dashboard.py
```

---

## 🐳 Déploiement Kubeflow sur Minikube

### Prérequis
- Docker Desktop
- Minikube v1.38+
- kubectl v1.34+
- 16 GB RAM minimum
- 40 GB espace disque

### Installation

```bash
# Démarrer Minikube
minikube start --driver=docker --cpus=4 --memory=6144 --disk-size=40g

# Installer Kubeflow Pipelines
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.0"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.0.0"

# Vérifier les pods
kubectl get pods -n kubeflow

# Accéder à l'interface (quand ml-pipeline-ui est Running)
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

### Exécution du pipeline dans Kubernetes

```bash
# Entrer dans le pod ml-pipeline
kubectl exec -it <ml-pipeline-pod-name> -n kubeflow -- bash

# Dans le pod
cd /app/smart_ecommerce
python kubeflow_pipeline.py --local
```

---

## 📈 Pipeline Kubeflow — 3 étapes

```
dataset_features.csv
        ↓
Étape 1 — Chargement et validation des données
        ↓
Étape 2 — Validation qualité (NaN, Top-K, clusters)
        ↓
Étape 3 — Export (CSV + Top-K + Anomalies + Rapport JSON)
```

---

## 🤖 Module LLM

Le dashboard intègre un module LLM (LLaMA 3.3 70B via Groq) qui permet de :

- **Résumer automatiquement** les Top-K produits par catégorie
- **Analyser la concurrence** entre les 5 boutiques
- **Répondre à des questions** sur les données via un chatbot BI

### Configuration

```python
# Dans dashboard.py
GROQ_API_KEY = "votre_cle_api_groq"
GROQ_MODEL   = "llama-3.3-70b-versatile"
```

---

## 📱 Dashboard — 6 onglets

| Onglet | Contenu |
|---|---|
| 📊 Vue générale | 6 KPIs + 4 graphiques interactifs |
| 🏆 Top-K Produits | Tableau filtrable + scatter + distribution |
| 🔵 Clustering ML | KMeans + Box plot + Scatter + PCA |
| ⚠️ Anomalies DBSCAN | Graphiques + PCA surlignée + tableau |
| 🔗 Règles d'association | Scatter support/confidence + filtres |
| 🤖 Intelligence LLM | Résumé + analyse concurrentielle + chatbot |

---

## 🗃️ Dataset

- **5 boutiques** : BeautyMarket, Justyol, Kiabi (Shopify) + Lasolda, Lhmiza (WooCommerce)
- **11 501 produits** après nettoyage
- **22 catégories**
- **2 453 produits Top-K** (21.3% du dataset)
- **8 anomalies** détectées par DBSCAN

---

## 📁 requirements.txt

```
selenium
webdriver-manager
beautifulsoup4
requests
pandas
numpy
scikit-learn
mlxtend
plotly
streamlit
groq
kfp==2.7.0
matplotlib
seaborn
xgboost
```

---

## 👥 Auteurs

- **Projet pédagogique** — FST Tanger
- **Filière** : LSI 2
- **Module** : DM & SID
- **Année** : 2025/2026

---

## 📄 Licence

Ce projet est réalisé dans un cadre pédagogique à la FST Tanger.
