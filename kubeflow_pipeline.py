"""
Smart eCommerce Intelligence — Pipeline Kubeflow (VERSION OPTIMISÉE)
=====================================================================
Pipeline qui UTILISE vos fichiers existants sans rien recalculer.

Fichiers utilisés :
- dataset_features.csv (déjà nettoyé + features ML + clusters + PCA + DBSCAN)

Exécution :
    python kubeflow_pipeline.py --local      # Tester localement (sans Kubeflow)
    python kubeflow_pipeline.py --compile    # Compiler en YAML pour Kubeflow ( genere smart_ecommerce_pipeline.yaml ) 


"""

import subprocess
import sys
import os
import json
from datetime import datetime

# ============================================================
# VÉRIFICATION DES DÉPENDANCES
# ============================================================
def installer_dependances():
    packages = ["kfp==2.7.0", "pandas", "numpy"]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import kfp
    from kfp import dsl
    from kfp.dsl import component, pipeline, Output, Input, Dataset, Metrics
    print(f"KFP version : {kfp.__version__}")
except ImportError:
    print("Installation de kfp...")
    installer_dependances()
    import kfp
    from kfp import dsl
    from kfp.dsl import component, pipeline, Output, Input, Dataset, Metrics


# ============================================================
# COMPOSANT 1 — CHARGEMENT dataset_features.csv
# ============================================================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy"]
)
def load_features_component(
    output_dataset: Output[Dataset],
    output_metrics: Output[Metrics]
):
    """
    Charge directement dataset_features.csv qui contient DÉJÀ :
    - Nettoyage
    - score_popularite
    - top_k
    - cluster (KMeans K=5)
    - cluster_nom
    - pca1, pca2
    - dbscan_label
    
    RIEN n'est recalculé !
    """
    import pandas as pd
    import os
    
    print("=" * 50)
    print("ÉTAPE 1 — CHARGEMENT dataset_features.csv")
    print("=" * 50)
    
    fichier = "dataset_features.csv"
    
    if not os.path.exists(fichier):
        raise FileNotFoundError(
            f" {fichier} non trouvé !\n"
            "Lance d'abord : python ml_pipeline_complet.py"
        )
    
    print(f" Chargement de {fichier}")
    df = pd.read_csv(fichier, encoding="utf-8-sig")
    
    # Statistiques
    print(f"\n Données chargées :")
    print(f"   - Produits total      : {len(df)}")
    print(f"   - Boutiques           : {df['shop'].nunique()}")
    print(f"   - Catégories          : {df['categorie'].nunique()}")
    
    if "top_k" in df.columns:
        print(f"   - Top-K produits      : {df['top_k'].sum()}")
    
    if "cluster_nom" in df.columns:
        print(f"   - Clusters KMeans     : {df['cluster_nom'].nunique()}")
        print(f"     {df['cluster_nom'].value_counts().to_dict()}")
    
    if "dbscan_label" in df.columns:
        anomalies = (df["dbscan_label"] == -1).sum()
        clusters = df[df["dbscan_label"] != -1]["dbscan_label"].nunique()
        print(f"   - Clusters DBSCAN     : {clusters}")
        print(f"   - Anomalies DBSCAN    : {anomalies}")
    
    if "pca1" in df.columns and "pca2" in df.columns:
        print(f"   - PCA                 : 2 composantes disponibles")
    
    # Sauvegarde pour l'étape suivante
    df.to_csv(output_dataset.path, index=False, encoding="utf-8-sig")
    
    # Métriques pour Kubeflow
    output_metrics.log_metric("produits_total", len(df))
    output_metrics.log_metric("nb_boutiques", int(df["shop"].nunique()))
    output_metrics.log_metric("nb_categories", int(df["categorie"].nunique()))
    
    if "top_k" in df.columns:
        output_metrics.log_metric("top_k_count", int(df["top_k"].sum()))
        output_metrics.log_metric("pct_topk", round(df["top_k"].mean() * 100, 2))
    
    if "prix" in df.columns:
        output_metrics.log_metric("prix_moyen", round(df["prix"].mean(), 2))
        output_metrics.log_metric("prix_median", round(df["prix"].median(), 2))
    
    if "score_popularite" in df.columns:
        output_metrics.log_metric("score_moyen", round(df["score_popularite"].mean(), 4))
    
    if "dbscan_label" in df.columns:
        output_metrics.log_metric("anomalies_dbscan", int((df["dbscan_label"] == -1).sum()))
    
    print(f"\n Données prêtes : {output_dataset.path}")


# ============================================================
# COMPOSANT 2 — VALIDATION (optionnel mais utile)
# ============================================================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy"]
)
def validation_component(
    input_dataset: Input[Dataset],
    output_metrics: Output[Metrics]
):
    """
    Valide la qualité du dataset_features.csv
    """
    import pandas as pd
    import numpy as np
    
    print("=" * 50)
    print("ÉTAPE 2 — VALIDATION QUALITÉ")
    print("=" * 50)
    
    df = pd.read_csv(input_dataset.path, engine="python")
    
    # Vérifications
    print(f"\n Vérifications :")
    
    # 1. Score popularité
    if "score_popularite" in df.columns:
        nan_score = df["score_popularite"].isna().sum()
        print(f"   - NaN score_popularite : {nan_score} / {len(df)}")
        output_metrics.log_metric("nan_score", int(nan_score))
        
        if nan_score > 0:
            print(f"       ALERTE : {nan_score} scores NaN")
    
    # 2. Top-K
    if "top_k" in df.columns:
        pct_topk = df["top_k"].mean() * 100
        print(f"   - % Top-K              : {pct_topk:.1f}%")
        output_metrics.log_metric("validation_pct_topk", round(pct_topk, 1))
        
        # Top-K par shop
        print(f"\n    Top-K par boutique :")
        topk_shop = df.groupby("shop")["top_k"].agg(["sum", "count"])
        topk_shop["%"] = (topk_shop["sum"] / topk_shop["count"] * 100).round(1)
        for shop, row in topk_shop.iterrows():
            print(f"      {shop}: {int(row['sum'])} / {int(row['count'])} ({row['%']:.1f}%)")
    
    # 3. Clusters KMeans
    if "cluster_nom" in df.columns:
        print(f"\n    Clusters KMeans :")
        for nom, count in df["cluster_nom"].value_counts().items():
            print(f"      {nom}: {count} produits")
        output_metrics.log_metric("n_clusters_kmeans", int(df["cluster_nom"].nunique()))
    
    # 4. DBSCAN
    if "dbscan_label" in df.columns:
        anomalies = (df["dbscan_label"] == -1).sum()
        clusters = df[df["dbscan_label"] != -1]["dbscan_label"].nunique()
        print(f"\n    DBSCAN : {clusters} clusters, {anomalies} anomalies")
        output_metrics.log_metric("validation_anomalies", int(anomalies))
    
    # 5. Prix
    if "prix" in df.columns:
        prix_min = df["prix"].min()
        prix_max = df["prix"].max()
        print(f"\n    Prix : min={prix_min:.0f} DH, max={prix_max:.0f} DH")
        
        if prix_min < 5:
            print(f"       ALERTE : prix minimum anormal ({prix_min:.2f} DH)")
        if prix_max > 50000:
            print(f"       ALERTE : prix maximum anormal ({prix_max:.0f} DH)")
    
    print("\n Validation terminée")


# ============================================================
# COMPOSANT 3 — EXPORT POUR KUBEFLOW
# ============================================================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "numpy"]
)
def export_component(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
    output_metrics: Output[Metrics]
):
    """
    Export final formaté pour Kubeflow Pipelines.
    Génère les artifacts requis.
    """
    import pandas as pd
    import json
    from datetime import datetime
    
    print("=" * 50)
    print("ÉTAPE 3 — EXPORT KUBEFLOW")
    print("=" * 50)
    
    df = pd.read_csv(input_dataset.path, engine="python")
    
    # 1. Dataset complet (passé à l'output_dataset)
    df.to_csv(output_dataset.path, index=False, encoding="utf-8-sig")
    print(f" Dataset complet : {output_dataset.path}")
    
    # 2. Top-K pour dashboard
    if "top_k" in df.columns:
        df_topk = df[df["top_k"] == 1].sort_values(
            "score_popularite", ascending=False
        )
        topk_path = output_dataset.path.replace(".csv", "_topk.csv")
        df_topk.to_csv(topk_path, index=False, encoding="utf-8-sig")
        print(f" Top-K produits  : {topk_path} ({len(df_topk)} produits)")
        output_metrics.log_metric("export_topk_count", len(df_topk))
    
    # 3. Anomalies
    if "dbscan_label" in df.columns:
        df_anomalies = df[df["dbscan_label"] == -1]
        if len(df_anomalies) > 0:
            anomalies_path = output_dataset.path.replace(".csv", "_anomalies.csv")
            df_anomalies.to_csv(anomalies_path, index=False, encoding="utf-8-sig")
            print(f" Anomalies       : {anomalies_path} ({len(df_anomalies)} anomalies)")
    
    # 4. Rapport JSON
    rapport = {
        "pipeline": "Smart eCommerce Intelligence",
        "version": "2.0.0",
        "date_execution": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "statistiques": {
            "produits_total": len(df),
            "boutiques": df["shop"].nunique(),
            "categories": df["categorie"].nunique(),
            "prix_moyen": round(df["prix"].mean(), 2) if "prix" in df.columns else None,
            "score_moyen": round(df["score_popularite"].mean(), 4) if "score_popularite" in df.columns else None,
            "top_k_count": int(df["top_k"].sum()) if "top_k" in df.columns else 0,
            "pct_topk": round(df["top_k"].mean() * 100, 1) if "top_k" in df.columns else 0,
            "anomalies_dbscan": int((df["dbscan_label"] == -1).sum()) if "dbscan_label" in df.columns else 0,
            "clusters_kmeans": int(df["cluster_nom"].nunique()) if "cluster_nom" in df.columns else 0,
        }
    }
    
    rapport_path = output_dataset.path.replace(".csv", "_rapport.json")
    with open(rapport_path, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2)
    print(f" Rapport JSON    : {rapport_path}")
    
    # Métriques finales
    output_metrics.log_metric("pipeline_success", 1)
    output_metrics.log_metric("total_artifacts", 4)
    
    print("\n" + "=" * 50)
    print(" PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 50)


# ============================================================
# DÉFINITION DU PIPELINE (3 étapes seulement)
# ============================================================
@pipeline(
    name="smart-ecommerce-pipeline-optimized",
    description="Pipeline optimisé — utilise dataset_features.csv existant"
)
def smart_ecommerce_pipeline():
    """
    Pipeline Kubeflow qui utilise vos données déjà préparées.
    3 étapes seulement : Charger → Valider → Exporter
    """
    
    # Étape 1 — Charger dataset_features.csv
    step1 = load_features_component()
    step1.set_display_name("1. Charger dataset_features.csv")
    
    # Étape 2 — Validation
    step2 = validation_component(
        input_dataset=step1.outputs["output_dataset"]
    )
    step2.set_display_name("2. Validation qualité")
    step2.after(step1)
    
    # Étape 3 — Export
    step3 = export_component(
        input_dataset=step1.outputs["output_dataset"]
    )
    step3.set_display_name("3. Export pour Kubeflow")
    step3.after(step2)


# ============================================================
# COMPILATION + EXÉCUTION LOCALE
# ============================================================
def compiler_pipeline():
    """Compile le pipeline en YAML pour Kubeflow."""
    from kfp import compiler
    output_file = "smart_ecommerce_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=smart_ecommerce_pipeline,
        package_path=output_file
    )
    print(f" Pipeline compilé → {output_file}")
    return output_file


def executer_localement():
    """Exécution locale du pipeline (test)."""
    import tempfile
    import shutil
    
    print("\n" + "=" * 55)
    print("EXÉCUTION LOCALE — PIPELINE OPTIMISÉ")
    print("=" * 55)
    print(f"Début : {datetime.now().strftime('%H:%M:%S')}")
    
    # Vérifier que dataset_features.csv existe
    if not os.path.exists("dataset_features.csv"):
        print("\n ERREUR : dataset_features.csv non trouvé !")
        print("Lance d'abord : python ml_pipeline_complet.py")
        return
    
    tmpdir = tempfile.mkdtemp()
    
    class FakeDataset:
        def __init__(self, path):
            self.path = path
    
    class FakeMetrics:
        def __init__(self):
            self.metrics = {}
        def log_metric(self, name, value):
            self.metrics[name] = value
            print(f"   {name} = {value}")
    
    path_input = os.path.join(tmpdir, "input.csv")
    path_output = os.path.join(tmpdir, "output.csv")
    
    print("\n[1/3] Chargement de dataset_features.csv...")
    load_features_component.python_func(
        output_dataset=FakeDataset(path_input),
        output_metrics=FakeMetrics()
    )
    
    print("\n[2/3] Validation...")
    validation_component.python_func(
        input_dataset=FakeDataset(path_input),
        output_metrics=FakeMetrics()
    )
    
    print("\n[3/3] Export...")
    export_component.python_func(
        input_dataset=FakeDataset(path_input),
        output_dataset=FakeDataset(path_output),
        output_metrics=FakeMetrics()
    )
    
    # Copier les résultats
    shutil.copy(path_output, "kubeflow_export.csv")
    
    topk_file = path_output.replace(".csv", "_topk.csv")
    if os.path.exists(topk_file):
        shutil.copy(topk_file, "kubeflow_topk.csv")
    
    anomalies_file = path_output.replace(".csv", "_anomalies.csv")
    if os.path.exists(anomalies_file):
        shutil.copy(anomalies_file, "kubeflow_anomalies.csv")
    
    rapport_file = path_output.replace(".csv", "_rapport.json")
    if os.path.exists(rapport_file):
        shutil.copy(rapport_file, "kubeflow_rapport.json")
    
    print("\n" + "=" * 55)
    print(" Fichiers générés :")
    print("   - kubeflow_export.csv")
    print("   - kubeflow_topk.csv")
    print("   - kubeflow_anomalies.csv")
    print("   - kubeflow_rapport.json")
    print("=" * 55)
    print(f"Fin : {datetime.now().strftime('%H:%M:%S')}")


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smart eCommerce Pipeline — Version optimisée"
    )
    parser.add_argument("--compile", action="store_true",
                        help="Compiler le pipeline en YAML")
    parser.add_argument("--local", action="store_true",
                        help="Exécution locale (test)")
    parser.add_argument("--upload", action="store_true",
                        help="Upload sur Kubeflow")
    args = parser.parse_args()
    
    if args.compile or (not args.local and not args.upload):
        compiler_pipeline()
    
    if args.local:
        executer_localement()
    
    if args.upload:
        try:
            client = kfp.Client(host="http://localhost:8080")
            client.create_run_from_pipeline_func(
                smart_ecommerce_pipeline,
                experiment_name="Smart eCommerce",
                run_name=f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            print(" Pipeline uploadé sur Kubeflow !")
        except Exception as e:
            print(f" Erreur upload : {e}")
            print("Kubeflow non accessible — utilise --local pour tester")