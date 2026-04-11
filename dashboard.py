import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from groq import Groq

# Clé API Groq — mets ta nouvelle clé ici après l'avoir régénérée
GROQ_API_KEY = "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"
GROQ_MODEL   = "openai/gpt-oss-120b"

@st.cache_data(show_spinner=False)
def appeler_llm(prompt: str) -> str:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        # PAR
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,   # ✅
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur LLM : {e}"


# ============================================================
# CONFIGURATION PAGE
# ============================================================
st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE CSS — Dégradé violet/bleu premium
# ============================================================
st.markdown("""
<style>
    /* Fond principal dégradé */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04) !important;
        border-right: 1px solid rgba(167,139,250,0.2) !important;
    }
    [data-testid="stSidebar"] * {
        color: rgba(255,255,255,0.85) !important;
    }

    /* Texte global */
    .stApp, .stApp p, .stApp label, .stApp div {
        color: rgba(255,255,255,0.85);
    }

    /* Métriques KPI */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(167,139,250,0.25);
        border-radius: 12px;
        padding: 16px !important;
        backdrop-filter: blur(10px);
    }
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.55) !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 26px !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #a78bfa !important;
    }

    /* Titres de section */
    h1 { color: #ffffff !important; font-size: 28px !important; }
    h2 { color: #a78bfa !important; font-size: 20px !important; }
    h3 { color: rgba(255,255,255,0.9) !important; font-size: 16px !important; }

    /* Cards dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }

    /* Selectbox et filtres */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stMultiSelect > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
        border-radius: 8px !important;
    }
    .stSlider > div {
        color: #a78bfa !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(167,139,250,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255,255,255,0.5);
        border-radius: 8px;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(167,139,250,0.2) !important;
        color: #a78bfa !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #7F77DD, #378ADD);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(167,139,250,0.2) !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.85) !important;
    }

    /* Info / success boxes */
    .stAlert {
        background: rgba(167,139,250,0.1) !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.85) !important;
    }

    /* Header badge */
    .header-badge {
        display: inline-block;
        background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.4);
        color: #a78bfa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin-left: 10px;
    }

    /* Score pills */
    .score-high { color: #5DCAA5; font-weight: 600; }
    .score-med  { color: #EF9F27; font-weight: 600; }
    .score-low  { color: #F09595; font-weight: 600; }

    /* Section card */
    .section-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# TRADUCTIONS FR / EN
# ============================================================
TEXTES = {
    "FR": {
        "titre":         "Smart eCommerce Intelligence",
        "sous_titre":    "Analyse ML · Top-K · Clustering · Anomalies",
        "langue":        "Langue / Language",
        "nav_overview":  "Vue générale",
        "nav_topk":      "Top-K Produits",
        "nav_ml":        "Clustering ML",
        "nav_anomalies": "Anomalies DBSCAN",
        "nav_rules":     "Règles d'association",
        "kpi_produits":  "Produits analysés",
        "kpi_topk":      "Top-K sélectionnés",
        "kpi_prix":      "Prix moyen (DH)",
        "kpi_anomalies": "Anomalies détectées",
        "kpi_shops":     "Boutiques",
        "kpi_categories":"Catégories",
        "filtre_shop":   "Filtrer par boutique",
        "filtre_cat":    "Filtrer par catégorie",
        "filtre_score":  "Score minimum",
        "filtre_prix":   "Fourchette de prix (DH)",
        "topk_titre":    "Meilleurs produits — Top-K",
        "col_produit":   "Produit",
        "col_shop":      "Boutique",
        "col_cat":       "Catégorie",
        "col_prix":      "Prix (DH)",
        "col_score":     "Score",
        "col_promo":     "Promo",
        "col_stock":     "Stock",
        "col_cluster":   "Segment",
        "cluster_titre": "Segmentation KMeans K=5",
        "pca_titre":     "Visualisation PCA",
        "dbscan_titre":  "Anomalies détectées par DBSCAN",
        "rules_titre":   "Règles d'association (Apriori)",
        "dist_prix":     "Distribution des prix",
        "topk_shop":     "Top-K par boutique",
        "topk_cat":      "Top-K par catégorie",
        "segment_prix":  "Segments de prix",
        "repartition":   "Répartition par boutique",
        "tous":          "Tous",
        "oui":           "Oui",
        "non":           "Non",
        "en_stock":      "En stock",
        "epuise":        "Épuisé",
    },
    "EN": {
        "titre":         "Smart eCommerce Intelligence",
        "sous_titre":    "ML Analysis · Top-K · Clustering · Anomalies",
        "langue":        "Language / Langue",
        "nav_overview":  "Overview",
        "nav_topk":      "Top-K Products",
        "nav_ml":        "ML Clustering",
        "nav_anomalies": "DBSCAN Anomalies",
        "nav_rules":     "Association Rules",
        "kpi_produits":  "Products analyzed",
        "kpi_topk":      "Top-K selected",
        "kpi_prix":      "Avg price (DH)",
        "kpi_anomalies": "Anomalies detected",
        "kpi_shops":     "Shops",
        "kpi_categories":"Categories",
        "filtre_shop":   "Filter by shop",
        "filtre_cat":    "Filter by category",
        "filtre_score":  "Minimum score",
        "filtre_prix":   "Price range (DH)",
        "topk_titre":    "Best products — Top-K",
        "col_produit":   "Product",
        "col_shop":      "Shop",
        "col_cat":       "Category",
        "col_prix":      "Price (DH)",
        "col_score":     "Score",
        "col_promo":     "Promo",
        "col_stock":     "Stock",
        "col_cluster":   "Segment",
        "cluster_titre": "KMeans K=5 Segmentation",
        "pca_titre":     "PCA Visualization",
        "dbscan_titre":  "DBSCAN Anomalies",
        "rules_titre":   "Association Rules (Apriori)",
        "dist_prix":     "Price distribution",
        "topk_shop":     "Top-K by shop",
        "topk_cat":      "Top-K by category",
        "segment_prix":  "Price segments",
        "repartition":   "Distribution by shop",
        "tous":          "All",
        "oui":           "Yes",
        "non":           "No",
        "en_stock":      "In stock",
        "epuise":        "Out of stock",
    }
}

# ============================================================
# COULEURS PLOTLY — thème violet/bleu premium
# ============================================================
COLORS = {
    "purple":  "#7F77DD",
    "blue":    "#378ADD",
    "teal":    "#1D9E75",
    "amber":   "#EF9F27",
    "coral":   "#D85A30",
    "pink":    "#D4537E",
    "green":   "#639922",
    "gray":    "#888780",
}

PALETTE_SHOPS = {
    "BeautyMarket": "#7F77DD",
    "Justyol":      "#378ADD",
    "Kiabi":        "#1D9E75",
    "Lasolda":      "#EF9F27",
    "Lhmiza":       "#D4537E",
}

PALETTE_CLUSTERS = {
    "Très low-cost": "#1D9E75",
    "Low-cost":      "#5DCAA5",
    "Mid-range":     "#378ADD",
    "Premium":       "#EF9F27",
    "Luxe":          "#D85A30",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="rgba(255,255,255,0.75)", size=12),
    title_font=dict(color="white", size=14),
    legend=dict(
        bgcolor="rgba(255,255,255,0.05)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        font=dict(color="rgba(255,255,255,0.75)")
    ),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        linecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color="rgba(255,255,255,0.55)")
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        linecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color="rgba(255,255,255,0.55)")
    ),
    margin=dict(t=40, b=30, l=30, r=20)
)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
@st.cache_data
def charger_donnees():
    df = pd.read_csv("dataset_features.csv", engine="python")

    # Nettoyage colonnes affichage
    df["prix"] = pd.to_numeric(df["prix"], errors="coerce")
    df["score_popularite"] = pd.to_numeric(df["score_popularite"], errors="coerce")
    df["promo_bin"] = pd.to_numeric(df["promo_bin"], errors="coerce").fillna(0)
    df["stock_bin"] = pd.to_numeric(df["stock_bin"], errors="coerce").fillna(0)

    # Labels lisibles
    df["promo_label"] = df["promo_bin"].apply(lambda x: "Oui" if x == 1 else "Non")
    df["stock_label"] = df["stock_bin"].apply(lambda x: "En stock" if x == 1 else "Épuisé")

    # cluster_nom si absent
    if "cluster_nom" not in df.columns and "cluster" in df.columns:
        prix_cluster = df.groupby("cluster")["prix"].mean()
        def nommer(p):
            if p < 100: return "Très low-cost"
            elif p < 200: return "Low-cost"
            elif p < 400: return "Mid-range"
            elif p < 800: return "Premium"
            else: return "Luxe"
        df["cluster_nom"] = df["cluster"].map(prix_cluster).apply(nommer)

    return df

@st.cache_data
def charger_regles():
    try:
        return pd.read_csv("regles_association.csv", engine="python")
    except:
        return pd.DataFrame()

@st.cache_data
def charger_anomalies():
    try:
        return pd.read_csv("anomalies_dbscan.csv", engine="python")
    except:
        return pd.DataFrame()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    langue = st.selectbox("🌐 Langue / Language", ["FR", "EN"])
    T = TEXTES[langue]

    st.markdown("---")
    st.markdown("### Filtres globaux")

    df_raw = charger_donnees()

    shops_dispo = [T["tous"]] + sorted(df_raw["shop"].unique().tolist())
    shop_sel = st.selectbox(T["filtre_shop"], shops_dispo)

    cats_dispo = [T["tous"]] + sorted(df_raw["categorie"].dropna().unique().tolist())
    cat_sel = st.selectbox(T["filtre_cat"], cats_dispo)

    prix_min = int(df_raw["prix"].min())
    prix_max = int(df_raw["prix"].max())
    prix_range = st.slider(
        T["filtre_prix"],
        min_value=prix_min,
        max_value=prix_max,
        value=(prix_min, prix_max),
        step=10
    )

    score_min = st.slider(T["filtre_score"], 0.0, 1.0, 0.0, 0.05)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:rgba(255,255,255,0.35);text-align:center'>
    Smart eCommerce Intelligence<br>
    FST Tanger · LSI 2 · 2025/2026
    </div>
    """, unsafe_allow_html=True)

T = TEXTES[langue]

# ============================================================
# APPLICATION DES FILTRES
# ============================================================
df = df_raw.copy()
if shop_sel != T["tous"]:
    df = df[df["shop"] == shop_sel]
if cat_sel != T["tous"]:
    df = df[df["categorie"] == cat_sel]
df = df[(df["prix"] >= prix_range[0]) & (df["prix"] <= prix_range[1])]
df = df[df["score_popularite"] >= score_min]

df_topk = df[df["top_k"] == 1].sort_values("score_popularite", ascending=False)

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div style='text-align:center;padding:10px 0 20px'>
  <h1 style='font-size:32px;font-weight:700;
     background:linear-gradient(90deg,#a78bfa,#60a5fa);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;
     margin-bottom:6px'>
    {T["titre"]}
  </h1>
  <p style='color:rgba(255,255,255,0.45);font-size:14px'>{T["sous_titre"]}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGATION PAR ONGLETS
# ============================================================
# PAR
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    f"📊 {T['nav_overview']}",
    f"🏆 {T['nav_topk']}",
    f"🔵 {T['nav_ml']}",
    f"⚠️ {T['nav_anomalies']}",
    f"🔗 {T['nav_rules']}",
    f"🤖 Intelligence LLM",
])

# ════════════════════════════════════════════════════════════
# TAB 1 — VUE GÉNÉRALE
# ════════════════════════════════════════════════════════════
with tab1:

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(T["kpi_produits"],  f"{len(df):,}")
    c2.metric(T["kpi_topk"],      f"{len(df_topk):,}",
              f"{len(df_topk)/len(df)*100:.1f}%" if len(df) > 0 else "")
    c3.metric(T["kpi_prix"],      f"{df['prix'].mean():.0f} DH")
    c4.metric(T["kpi_anomalies"],
              str(int((df["dbscan_label"] == -1).sum())) if "dbscan_label" in df.columns else "N/A")
    c5.metric(T["kpi_shops"],     str(df["shop"].nunique()))
    c6.metric(T["kpi_categories"],str(df["categorie"].nunique()))

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Top-K par boutique
        topk_shop = df_topk.groupby("shop").size().reset_index(name="count")
        topk_shop = topk_shop.sort_values("count", ascending=True)
        fig = go.Figure(go.Bar(
            x=topk_shop["count"],
            y=topk_shop["shop"],
            orientation="h",
            marker=dict(
                color=[PALETTE_SHOPS.get(s, "#888") for s in topk_shop["shop"]],
                line=dict(width=0)
            ),
            text=topk_shop["count"],
            textposition="outside",
            textfont=dict(color="white", size=11)
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title=T["topk_shop"], height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Distribution des prix
        fig2 = go.Figure()
        for shop in df["shop"].unique():
            sub = df[df["shop"] == shop]["prix"]
            fig2.add_trace(go.Histogram(
                x=sub, name=shop,
                marker_color=PALETTE_SHOPS.get(shop, "#888"),
                opacity=0.7, nbinsx=40
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, title=T["dist_prix"],
                           barmode="overlay", height=280)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Répartition produits par boutique (donut)
        shop_counts = df.groupby("shop").size().reset_index(name="nb")
        fig3 = go.Figure(go.Pie(
            labels=shop_counts["shop"],
            values=shop_counts["nb"],
            hole=0.55,
            marker=dict(colors=[PALETTE_SHOPS.get(s, "#888") for s in shop_counts["shop"]]),
            textfont=dict(color="white"),
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, title=T["repartition"], height=280)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Top-K par catégorie (top 10)
        topk_cat = df_topk.groupby("categorie").size().reset_index(name="count")
        topk_cat = topk_cat.nlargest(10, "count").sort_values("count", ascending=True)
        fig4 = go.Figure(go.Bar(
            x=topk_cat["count"],
            y=topk_cat["categorie"],
            orientation="h",
            marker=dict(
                color=COLORS["purple"],
                opacity=0.85,
                line=dict(width=0)
            ),
            text=topk_cat["count"],
            textposition="outside",
            textfont=dict(color="white", size=10)
        ))
        fig4.update_layout(**PLOTLY_LAYOUT, title=T["topk_cat"], height=280)
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — TOP-K PRODUITS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f"## 🏆 {T['topk_titre']}")

    # Filtres supplémentaires
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        nb_afficher = st.selectbox("Afficher / Show", [25, 50, 100, 200], index=0)
    with col_f2:
        tri_col = st.selectbox("Trier par / Sort by",
                               [T["col_score"], T["col_prix"]])
    with col_f3:
        tri_asc = st.selectbox("Ordre / Order", ["Décroissant ↓", "Croissant ↑"])

    tri_ascending = (tri_asc == "Croissant ↑")
    col_tri = "score_popularite" if tri_col == T["col_score"] else "prix"

    df_display = df_topk.sort_values(col_tri, ascending=tri_ascending).head(nb_afficher)

    # Score moyen Top-K
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Score moyen", f"{df_topk['score_popularite'].mean():.3f}")
    col_s2.metric("Prix moyen Top-K", f"{df_topk['prix'].mean():.0f} DH")
    col_s3.metric("En promo", f"{df_topk['promo_bin'].sum():.0f}")
    col_s4.metric("En stock", f"{df_topk['stock_bin'].sum():.0f}")

    st.markdown("---")

    # Tableau Top-K
    cols_afficher = {
        "produit":          T["col_produit"],
        "shop":             T["col_shop"],
        "categorie":        T["col_cat"],
        "prix":             T["col_prix"],
        "score_popularite": T["col_score"],
        "promo_label":      T["col_promo"],
        "stock_label":      T["col_stock"],
        "cluster_nom":      T["col_cluster"],
    }
    cols_dispo = [c for c in cols_afficher.keys() if c in df_display.columns]
    df_show = df_display[cols_dispo].rename(columns=cols_afficher)
    df_show[T["col_score"]] = df_show[T["col_score"]].round(4)
    df_show[T["col_prix"]]  = df_show[T["col_prix"]].round(0).astype(int)

    st.dataframe(
        df_show,
        use_container_width=True,
        height=420,
        hide_index=True
    )

    # Score distribution Top-K vs Non Top-K
    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=df[df["top_k"] == 1]["score_popularite"],
            name="Top-K", marker_color=COLORS["purple"],
            opacity=0.75, nbinsx=30
        ))
        fig_score.add_trace(go.Histogram(
            x=df[df["top_k"] == 0]["score_popularite"],
            name="Non Top-K", marker_color=COLORS["gray"],
            opacity=0.5, nbinsx=30
        ))
        fig_score.update_layout(**PLOTLY_LAYOUT,
                                title="Distribution scores — Top-K vs Non Top-K",
                                barmode="overlay", height=280)
        st.plotly_chart(fig_score, use_container_width=True)

    with col_g2:
        fig_scatter = go.Figure()
        for shop in df_topk["shop"].unique():
            sub = df_topk[df_topk["shop"] == shop].head(60)
            fig_scatter.add_trace(go.Scatter(
                x=sub["prix"],
                y=sub["score_popularite"],
                mode="markers",
                name=shop,
                marker=dict(
                    color=PALETTE_SHOPS.get(shop, "#888"),
                    size=7,
                    opacity=0.75
                ),
                text=sub["produit"],
                hovertemplate="<b>%{text}</b><br>Prix: %{x} DH<br>Score: %{y}<extra></extra>"
            ))
        fig_scatter.update_layout(**PLOTLY_LAYOUT,
                                title="Prix vs Score — Top-K produits",
                                height=280)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — CLUSTERING ML
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f"## 🔵 {T['cluster_titre']}")

    if "cluster_nom" in df.columns:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Clusters", df["cluster_nom"].nunique())
        col_m2.metric("Silhouette Score", "0.349")
        col_m3.metric("Algorithme", "KMeans K=5")

        st.markdown("---")
        col_k1, col_k2 = st.columns(2)

        with col_k1:
            # Profil des clusters
            profil = df.groupby("cluster_nom").agg(
                nb_produits=("produit", "count"),
                prix_moyen=("prix", "mean"),
                score_moyen=("score_popularite", "mean"),
                promo_pct=("promo_bin", "mean")
            ).round(2).reset_index()

            fig_clusters = go.Figure()
            for _, row in profil.iterrows():
                fig_clusters.add_trace(go.Bar(
                    name=row["cluster_nom"],
                    x=[row["cluster_nom"]],
                    y=[row["nb_produits"]],
                    marker_color=PALETTE_CLUSTERS.get(row["cluster_nom"], "#888"),
                    text=[f'{row["nb_produits"]}<br>{row["prix_moyen"]:.0f} DH'],
                    textposition="outside",
                    textfont=dict(color="white", size=10)
                ))
            fig_clusters.update_layout(**PLOTLY_LAYOUT,
                                       title="Nombre de produits par segment",
                                       showlegend=False, height=300)
            st.plotly_chart(fig_clusters, use_container_width=True)

        # PAR
        with col_k2:
            fig_box = go.Figure()
            for cluster in df["cluster_nom"].unique():
                sub = df[df["cluster_nom"] == cluster]["prix"]
                fig_box.add_trace(go.Box(
                    y=sub, name=cluster,
                    marker_color=PALETTE_CLUSTERS.get(cluster, "#888"),
                    line_color=PALETTE_CLUSTERS.get(cluster, "#888"),
                    fillcolor=PALETTE_CLUSTERS.get(cluster, "#888"),
                    opacity=0.7,
                    boxpoints=False
                ))
            fig_box.update_layout(**PLOTLY_LAYOUT,
                                title="Distribution prix par segment",
                                showlegend=False, height=300)
            st.plotly_chart(fig_box, use_container_width=True)

        # Scatter plot KMeans (log_prix vs score)
        # PAR
        if "log_prix" in df.columns:
            fig_km = go.Figure()
            df_sample = df.sample(min(3000, len(df)))
            for cluster in df_sample["cluster_nom"].unique():
                sub = df_sample[df_sample["cluster_nom"] == cluster]
                fig_km.add_trace(go.Scatter(
                    x=sub["log_prix"], y=sub["score_popularite"],
                    mode="markers", name=cluster,
                    marker=dict(color=PALETTE_CLUSTERS.get(cluster, "#888"),
                            size=5, opacity=0.6),
                    text=sub["produit"],
                    hovertemplate="<b>%{text}</b><br>Log Prix: %{x:.2f}<br>Score: %{y:.3f}<extra></extra>"
                ))
            fig_km.update_layout(**PLOTLY_LAYOUT,
                                title="KMeans — Log Prix vs Score de popularité",
                                height=350)
            st.plotly_chart(fig_km, use_container_width=True)

        # Tableau profil
        st.markdown("#### Profil détaillé des clusters")
        profil_display = profil.rename(columns={
            "cluster_nom": "Segment",
            "nb_produits": "Nb produits",
            "prix_moyen":  "Prix moyen (DH)",
            "score_moyen": "Score moyen",
            "promo_pct":   "% Promo"
        })
        profil_display["% Promo"] = (profil_display["% Promo"] * 100).round(1)
        st.dataframe(profil_display, use_container_width=True, hide_index=True)

    # PCA
    st.markdown("---")
    st.markdown(f"### 📐 {T['pca_titre']}")

    if "pca1" in df.columns and "pca2" in df.columns:
        col_p1, col_p2 = st.columns(2)

                # PAR
        with col_p1:
            fig_pca1 = go.Figure()
            df_pca_sample = df.sample(min(3000, len(df)))
            for cluster in df_pca_sample["cluster_nom"].unique():
                sub = df_pca_sample[df_pca_sample["cluster_nom"] == cluster]
                fig_pca1.add_trace(go.Scatter(
                    x=sub["pca1"], y=sub["pca2"],
                    mode="markers", name=cluster,
                    marker=dict(color=PALETTE_CLUSTERS.get(cluster, "#888"),
                            size=4, opacity=0.6),
                    text=sub["produit"],
                    hovertemplate="<b>%{text}</b><extra></extra>"
                ))
            fig_pca1.update_layout(**PLOTLY_LAYOUT,
                                title="PCA — coloré par segment KMeans",
                                height=320)
            st.plotly_chart(fig_pca1, use_container_width=True)

        # PAR
        with col_p2:
            fig_pca2 = go.Figure()
            df_pca_sample2 = df.sample(min(3000, len(df)))
            for shop in df_pca_sample2["shop"].unique():
                sub = df_pca_sample2[df_pca_sample2["shop"] == shop]
                fig_pca2.add_trace(go.Scatter(
                    x=sub["pca1"], y=sub["pca2"],
                    mode="markers", name=shop,
                    marker=dict(color=PALETTE_SHOPS.get(shop, "#888"),
                            size=4, opacity=0.6),
                    text=sub["produit"],
                    hovertemplate="<b>%{text}</b><extra></extra>"
                ))
            fig_pca2.update_layout(**PLOTLY_LAYOUT,
                                title="PCA — coloré par boutique",
                                height=320)
            st.plotly_chart(fig_pca2, use_container_width=True)

        st.info("PC1 (38.7%) = axe promotion/shop · PC2 (21.3%) = axe popularité/remise · Variance totale : 60.0%")
    else:
        st.warning("Colonnes PCA non trouvées. Lance d'abord ml_pipeline_complet.py")

# ════════════════════════════════════════════════════════════
# TAB 4 — ANOMALIES DBSCAN
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f"## ⚠️ {T['dbscan_titre']}")

    if "dbscan_label" in df.columns:
        anomalies_df = df[df["dbscan_label"] == -1].copy()
        normaux_df   = df[df["dbscan_label"] != -1].copy()

        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        col_d1.metric("Anomalies totales", len(anomalies_df))
        col_d2.metric("% du dataset", f"{len(anomalies_df)/len(df)*100:.2f}%")
        col_d3.metric("Prix moyen anomalies", f"{anomalies_df['prix'].mean():.0f} DH" if len(anomalies_df) > 0 else "N/A")
        col_d4.metric("Clusters DBSCAN", df["dbscan_label"].nunique() - 1)

        st.markdown("---")

        if len(anomalies_df) > 0:
            col_db1, col_db2 = st.columns(2)

            # PAR
            with col_db1:
                anom_shop = anomalies_df.groupby("shop").size().reset_index(name="nb")
                fig_anom = go.Figure()
                fig_anom.add_trace(go.Bar(
                    x=anom_shop["shop"],
                    y=anom_shop["nb"],
                    marker=dict(
                        color=[PALETTE_SHOPS.get(s, "#888") for s in anom_shop["shop"]],
                        line=dict(width=0)
                    ),
                    text=anom_shop["nb"],
                    textposition="outside",
                    textfont=dict(color="white", size=11)
                ))
                fig_anom.update_layout(**PLOTLY_LAYOUT,
                                    title="Anomalies par boutique",
                                    showlegend=False, height=280)
                st.plotly_chart(fig_anom, use_container_width=True)

            with col_db2:
                # Prix normaux vs anomalies
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Histogram(
                    x=normaux_df["prix"].clip(upper=normaux_df["prix"].quantile(0.99)),
                    name="Normaux", marker_color=COLORS["blue"],
                    opacity=0.6, nbinsx=40
                ))
                fig_comp.add_trace(go.Histogram(
                    x=anomalies_df["prix"],
                    name="Anomalies", marker_color=COLORS["coral"],
                    opacity=0.9, nbinsx=20
                ))
                fig_comp.update_layout(**PLOTLY_LAYOUT,
                                       title="Prix : normaux vs anomalies",
                                       barmode="overlay", height=280)
                st.plotly_chart(fig_comp, use_container_width=True)

            # PCA avec anomalies surlignées
            if "pca1" in df.columns:
                fig_pca_anom = go.Figure()
                fig_pca_anom.add_trace(go.Scatter(
                    x=normaux_df.sample(min(2000, len(normaux_df)))["pca1"],
                    y=normaux_df.sample(min(2000, len(normaux_df)))["pca2"],
                    mode="markers",
                    marker=dict(color=COLORS["blue"], size=4, opacity=0.3),
                    name="Produits normaux"
                ))
                fig_pca_anom.add_trace(go.Scatter(
                    x=anomalies_df["pca1"],
                    y=anomalies_df["pca2"],
                    mode="markers",
                    marker=dict(color=COLORS["coral"], size=12,
                               symbol="x", opacity=1.0,
                               line=dict(width=2, color="white")),
                    name="Anomalies",
                    text=anomalies_df["produit"],
                    hoverinfo="text"
                ))
                fig_pca_anom.update_layout(**PLOTLY_LAYOUT,
                                          title="PCA — Anomalies DBSCAN surlignées",
                                          height=350)
                st.plotly_chart(fig_pca_anom, use_container_width=True)

            # Tableau anomalies
            st.markdown("#### Détail des anomalies détectées")
            cols_anom = [c for c in ["shop", "categorie", "produit", "prix",
                                     "score_popularite", "remise_norm"] if c in anomalies_df.columns]
            st.dataframe(
                anomalies_df[cols_anom].sort_values("prix", ascending=False).round(3),
                use_container_width=True, hide_index=True
            )
        else:
            st.success("Aucune anomalie détectée avec les filtres actuels.")

    else:
        # Charger depuis le CSV anomalies
        anomalies_csv = charger_anomalies()
        if not anomalies_csv.empty:
            st.dataframe(anomalies_csv, use_container_width=True, hide_index=True)
        else:
            st.warning("Lance ml_pipeline_complet.py pour générer les anomalies DBSCAN.")

# ════════════════════════════════════════════════════════════
# TAB 5 — RÈGLES D'ASSOCIATION
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"## 🔗 {T['rules_titre']}")

    rules = charger_regles()

    if not rules.empty:
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Règles trouvées", len(rules))
        col_r2.metric("Lift max", f"{rules['lift'].max():.2f}")
        col_r3.metric("Confidence max", f"{rules['confidence'].max():.2f}")

        st.markdown("---")

        # Filtres règles
        col_rf1, col_rf2, col_rf3 = st.columns(3)
        with col_rf1:
            min_support = st.slider("Support minimum", 0.0, 1.0, 0.2, 0.05)
        with col_rf2:
            min_conf = st.slider("Confidence minimum", 0.0, 1.0, 0.5, 0.05)
        with col_rf3:
            min_lift = st.slider("Lift minimum", 1.0, 10.0, 1.0, 0.5)

        rules_filtered = rules[
            (rules["support"] >= min_support) &
            (rules["confidence"] >= min_conf) &
            (rules["lift"] >= min_lift)
        ].sort_values("lift", ascending=False)

        st.markdown(f"**{len(rules_filtered)} règles** après filtrage")

        col_rv1, col_rv2 = st.columns(2)

        with col_rv1:
            # Scatter support vs confidence
            fig_rules = px.scatter(
                rules_filtered.head(200),
                x="support", y="confidence",
                size="lift",
                color="lift",
                color_continuous_scale=[[0, "#378ADD"], [0.5, "#7F77DD"], [1, "#D4537E"]],
                hover_data=["antecedents", "consequents"],
                title="Support vs Confidence (taille = Lift)"
            )
            fig_rules.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_rules, use_container_width=True)

        with col_rv2:
            # Distribution du lift
            fig_lift = px.histogram(
                rules_filtered, x="lift",
                nbins=30,
                color_discrete_sequence=[COLORS["purple"]],
                title="Distribution du Lift"
            )
            fig_lift.update_layout(**PLOTLY_LAYOUT, height=320)
            st.plotly_chart(fig_lift, use_container_width=True)

        # Tableau des règles
        st.markdown("#### Top règles")
        rules_display = rules_filtered.head(50).copy()
        rules_display["antecedents"] = rules_display["antecedents"].astype(str).str.replace("frozenset", "").str.replace("{","").str.replace("}","").str.replace("'","")
        rules_display["consequents"] = rules_display["consequents"].astype(str).str.replace("frozenset", "").str.replace("{","").str.replace("}","").str.replace("'","")
        rules_display = rules_display[["antecedents", "consequents", "support", "confidence", "lift"]].round(3)
        rules_display.columns = ["Si (antécédent)", "Alors (conséquent)", "Support", "Confidence", "Lift"]
        st.dataframe(rules_display, use_container_width=True, hide_index=True)

    else:
        st.warning("Lance ml_pipeline_complet.py pour générer regles_association.csv")

# ════════════════════════════════════════════════════════════
# TAB 6 — MODULE LLM
# ════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## 🤖 Intelligence Augmentée par LLM")
    st.markdown("*Powered by openai/gpt-oss-120b via Groq*")

    # ── Section 1 : Résumé automatique Top-K ────────────────
    st.markdown("---")
    st.markdown("### Résumé automatique des Top-K produits")

    col_llm1, col_llm2 = st.columns(2)

    with col_llm1:
        categorie_llm = st.selectbox(
            "Choisir une catégorie à analyser",
            options=["Toutes"] + sorted(df_topk["categorie"].dropna().unique().tolist()),
            key="llm_cat"
        )
        nb_produits_llm = st.slider("Nombre de produits à analyser", 3, 15, 5, key="llm_nb")

    with col_llm2:
        shop_llm = st.selectbox(
            "Choisir une boutique",
            options=["Toutes"] + sorted(df_topk["shop"].unique().tolist()),
            key="llm_shop"
        )

    # Filtrage pour le LLM
    df_llm = df_topk.copy()
    if categorie_llm != "Toutes":
        df_llm = df_llm[df_llm["categorie"] == categorie_llm]
    if shop_llm != "Toutes":
        df_llm = df_llm[df_llm["shop"] == shop_llm]
    df_llm = df_llm.head(nb_produits_llm)

    if st.button("Générer le résumé Top-K", key="btn_resume"):
        if len(df_llm) == 0:
            st.warning("Aucun produit trouvé avec ces filtres.")
        else:
            # Construction du prompt
            produits_str = ""
            for _, row in df_llm.iterrows():
                produits_str += (
                    f"- {row['produit']} | Shop: {row['shop']} | "
                    f"Prix: {row['prix']:.0f} DH | "
                    f"Score: {row['score_popularite']:.3f} | "
                    f"Catégorie: {row['categorie']} | "
                    f"Promo: {'Oui' if row['promo_bin'] == 1 else 'Non'}\n"
                )

            prompt_resume = f"""Tu es un expert en e-commerce marocain. 
Voici les {nb_produits_llm} meilleurs produits (Top-K) détectés par notre système d'analyse ML :

{produits_str}

Génère un résumé analytique professionnel en français qui :
1. Identifie les tendances communes entre ces produits
2. Explique pourquoi ces produits sont populaires
3. Donne 3 recommandations business concrètes
4. Utilise un ton professionnel et concis (max 300 mots)"""

            with st.spinner("Le LLM analyse les produits..."):
                resume = appeler_llm(prompt_resume)

            st.markdown("#### Analyse générée :")
            st.markdown(f"""
            <div style='background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.3);
                        border-radius:12px;padding:20px;color:rgba(255,255,255,0.85);
                        line-height:1.7;font-size:14px'>
            {resume.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

    # ── Section 2 : Analyse concurrentielle ─────────────────
    st.markdown("---")
    st.markdown("### Analyse concurrentielle entre boutiques")

    if st.button("Générer l'analyse concurrentielle", key="btn_concurrence"):
        # Stats par shop
        stats_shops = df.groupby("shop").agg(
            nb_produits=("produit", "count"),
            prix_moyen=("prix", "mean"),
            score_moyen=("score_popularite", "mean"),
            pct_promo=("promo_bin", "mean"),
            nb_topk=("top_k", "sum")
        ).round(2).reset_index()

        stats_str = ""
        for _, row in stats_shops.iterrows():
            stats_str += (
                f"- {row['shop']} : {int(row['nb_produits'])} produits | "
                f"Prix moyen {row['prix_moyen']:.0f} DH | "
                f"Score moyen {row['score_moyen']:.3f} | "
                f"Promo {row['pct_promo']*100:.0f}% | "
                f"Top-K : {int(row['nb_topk'])} produits\n"
            )

        prompt_concurrence = f"""Tu es un analyste e-commerce spécialisé dans le marché marocain.
Voici les statistiques de {len(stats_shops)} boutiques en ligne marocaines :

{stats_str}

Génère une analyse concurrentielle professionnelle en français qui :
1. Compare les forces et faiblesses de chaque boutique
2. Identifie le leader du marché et explique pourquoi
3. Détecte les niches de marché exploitées
4. Propose 3 stratégies pour améliorer le positionnement
Sois concis et analytique (max 350 mots)."""

        with st.spinner("Analyse concurrentielle en cours..."):
            analyse_conc = appeler_llm(prompt_concurrence)

        st.markdown("#### Analyse concurrentielle :")
        st.markdown(f"""
        <div style='background:rgba(55,138,221,0.08);border:1px solid rgba(55,138,221,0.3);
                    border-radius:12px;padding:20px;color:rgba(255,255,255,0.85);
                    line-height:1.7;font-size:14px'>
        {analyse_conc.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    # ── Section 3 : Chatbot BI ───────────────────────────────
    st.markdown("---")
    st.markdown("### Chatbot BI — Pose une question sur les données")

    # Initialisation historique chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Affichage historique
    for msg in st.session_state.chat_history:
        role_color = "#7F77DD" if msg["role"] == "user" else "#1D9E75"
        role_label = "Toi" if msg["role"] == "user" else "LLM"
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.04);border-left:3px solid {role_color};
                    border-radius:8px;padding:12px 16px;margin:8px 0;
                    color:rgba(255,255,255,0.85);font-size:13px'>
        <strong style='color:{role_color}'>{role_label} :</strong><br>
        {msg["content"].replace(chr(10), "<br>")}
        </div>
        """, unsafe_allow_html=True)

    # Suggestions de questions
    st.markdown("**Suggestions :**")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        if st.button("Quels sont les produits les plus populaires ?", key="sug1"):
            st.session_state.question_input = "Quels sont les produits les plus populaires ?"
    with col_s2:
        if st.button("Analyse les tendances de prix", key="sug2"):
            st.session_state.question_input = "Analyse les tendances de prix par catégorie"
    with col_s3:
        if st.button("Quelles promotions sont efficaces ?", key="sug3"):
            st.session_state.question_input = "Quelles promotions sont les plus efficaces ?"

    # Input question
    question = st.text_input(
        "Pose ta question ici...",
        key="chat_input",
        placeholder="Ex: Quels sont les 5 produits émergents cette semaine ?"
    )

    if st.button("Envoyer", key="btn_chat") and question:
        # Contexte données pour le LLM
        contexte = f"""Tu es un assistant BI expert en e-commerce marocain.
Voici le contexte des données analysées :
- Dataset : {len(df)} produits de {df['shop'].nunique()} boutiques
- Boutiques : {', '.join(df['shop'].unique())}
- Catégories : {df['categorie'].nunique()} catégories
- Prix moyen : {df['prix'].mean():.0f} DH
- Top-K produits : {df['top_k'].sum()} produits sélectionnés
- Produits en promo : {df['promo_bin'].sum()} produits
- Score moyen : {df['score_popularite'].mean():.3f}

Top 5 produits actuels :
"""
        for _, row in df_topk.head(5).iterrows():
            contexte += f"- {row['produit']} ({row['shop']}) : {row['prix']:.0f} DH, score {row['score_popularite']:.3f}\n"

        prompt_chat = f"{contexte}\n\nQuestion de l'utilisateur : {question}\n\nRéponds en français, de façon claire et concise (max 200 mots)."

        # Ajouter question à l'historique
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Réflexion en cours..."):
            reponse = appeler_llm(prompt_chat)

        # Ajouter réponse à l'historique
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reponse
        })

        st.rerun()

    # Bouton reset chat
    if st.session_state.chat_history:
        if st.button("Effacer la conversation", key="btn_reset"):
            st.session_state.chat_history = []
            st.rerun()