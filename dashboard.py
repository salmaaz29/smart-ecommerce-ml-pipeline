# dashboard.py — Smart eCommerce Intelligence
# Lance avec : streamlit run dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Smart eCommerce Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0a;
    color: #e8e6e0;
}
.stApp { background-color: #0a0a0a; }
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}
[data-testid="stSidebar"] * { color: #888 !important; font-size: 11px !important; }
[data-testid="stSidebar"] .sidebar-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    color: #fff !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #222;
    border-radius: 4px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #555 !important; font-size: 10px !important; letter-spacing: 0.15em; text-transform: uppercase; }
[data-testid="stMetricValue"] { color: #e8e6e0 !important; font-family: 'Syne', sans-serif !important; font-size: 1.8rem !important; }
.section-label { font-size: 9px; letter-spacing: 0.25em; color: #444; text-transform: uppercase; margin-bottom: 4px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.6rem; font-weight: 700; color: #e8e6e0; margin-bottom: 20px; line-height: 1.2; }
.card { background: #111; border: 1px solid #1e1e1e; border-radius: 6px; padding: 20px; margin-bottom: 12px; }
.card-label { font-size: 9px; letter-spacing: 0.2em; color: #444; text-transform: uppercase; margin-bottom: 8px; }
.card-value { font-family: 'Syne', sans-serif; font-size: 2rem; color: #e8e6e0; font-weight: 700; }
.card-sub { font-size: 11px; color: #555; margin-top: 4px; }
.tag { display: inline-block; background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 2px; padding: 2px 8px; font-size: 10px; color: #666; margin: 2px; letter-spacing: 0.1em; }
.accent { color: #c8a97e; }
.divider { border: none; border-top: 1px solid #1e1e1e; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ── CHARGEMENT DATA ───────────────────────────────────────────
@st.cache_data
def load():
    df = pd.read_csv("dataset_clusters.csv", engine="python")
    try:
        rules = pd.read_csv("regles_association.csv")
    except:
        rules = pd.DataFrame()
    return df, rules

df, rules = load()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">◈ Smart eCommerce</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Intelligence Pipeline</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "Overview", "Top-K Products", "Price Analysis",
        "ML Clustering", "Random Forest", "Apriori Rules", "LLM Insights"
    ], label_visibility="collapsed")

    st.markdown("---")

    # ✅ Shop en premier
    shop_f = st.selectbox("Shop", ["All"] + sorted(df["shop"].unique().tolist()))

    # ✅ Catégories dynamiques selon shop
    if shop_f != "All":
        cats_dispo = sorted(df[df["shop"] == shop_f]["categorie"].dropna().unique().tolist())
    else:
        cats_dispo = sorted(df["categorie"].dropna().unique().tolist())
    cat_f = st.selectbox("Category", ["All"] + cats_dispo)

    # ✅ Segments dynamiques selon shop + catégorie
    df_temp = df.copy()
    if shop_f != "All": df_temp = df_temp[df_temp["shop"] == shop_f]
    if cat_f  != "All": df_temp = df_temp[df_temp["categorie"] == cat_f]
    segs_dispo = sorted(df_temp["segment_prix"].dropna().unique().tolist())
    seg_f = st.selectbox("Segment", ["All"] + segs_dispo)

    st.markdown("---")
    st.markdown('<div style="font-size:9px;color:#333;letter-spacing:0.1em">ARCHITECTURE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:9px;color:#444">MCP Host / Client / Server</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:9px;color:#444">{len(df):,} produits chargés</div>', unsafe_allow_html=True)

# ✅ Application des filtres
dff = df.copy()
if shop_f != "All": dff = dff[dff["shop"] == shop_f]
if cat_f  != "All": dff = dff[dff["categorie"] == cat_f]
if seg_f  != "All": dff = dff[dff["segment_prix"] == seg_f]

# ✅ Arrêt propre si vide
if len(dff) == 0:
    st.warning("⚠️ Aucun produit ne correspond à cette combinaison de filtres.")
    st.stop()

# Plotly theme
PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0a0a",
    plot_bgcolor="#0a0a0a",
    font=dict(family="DM Mono", color="#666", size=11),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="#1a1a1a", linecolor="#222", tickfont=dict(color="#555")),
    yaxis=dict(gridcolor="#1a1a1a", linecolor="#222", tickfont=dict(color="#555")),
    title_font=dict(family="Syne", color="#888", size=13),
)
ACCENT = "#c8a97e"
COLORS = ["#c8a97e","#7e9cc8","#7ec89c","#c87e9c","#9c7ec8"]

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<div class="section-label">Overview / Validated Pipeline State</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Un système intelligent d\'analyse<br>du marché e-commerce marocain.</div>', unsafe_allow_html=True)

    n        = len(dff)
    nb_shops = dff["shop"].nunique()
    nb_cats  = dff["categorie"].nunique()

    pct_prix   = (dff["prix"] > 0).sum()   / n * 100 if n > 0 else 0
    pct_rating = (dff["rating"] > 0).sum() / n * 100 if n > 0 else 0
    pct_stock  = dff["stock_bin"].sum()     / n * 100 if n > 0 else 0
    pct_promo  = dff["promo_bin"].sum()     / n * 100 if n > 0 else 0

    st.markdown(f"""
    <div class="card">
        <div class="card-label">Lecture du marché</div>
        <div style="font-size:13px;color:#666;line-height:1.8">
            Le pipeline couvre <span class="accent">{n:,} produits</span> répartis sur
            <span class="accent">{nb_shops} boutique(s)</span> et
            <span class="accent">{nb_cats} catégorie(s)</span>.
            La plateforme dominante est <span class="accent">Shopify</span> avec Justyol en tête du volume.
            Le dataset contient <span class="accent">{int(dff["promo_bin"].sum()):,} produits en promotion</span>
            et un taux de disponibilité de <span class="accent">{pct_stock:.1f}%</span>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="card"><div class="card-label">Price Coverage</div><div class="card-value">{pct_prix:.1f}%</div><div class="card-sub">Produits avec prix valide</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="card-label">Rating Coverage</div><div class="card-value">{pct_rating:.1f}%</div><div class="card-sub">Produits avec note disponible</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="card-label">In-Stock Signal</div><div class="card-value">{pct_stock:.1f}%</div><div class="card-sub">Produits marqués disponibles</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><div class="card-label">Discounted Rows</div><div class="card-value">{pct_promo:.1f}%</div><div class="card-sub">Produits avec remise détectée</div></div>', unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        shop_counts = dff.groupby(["shop","platform"]).size().reset_index(name="n")
        fig = px.bar(shop_counts, x="shop", y="n", color="platform",
                     color_discrete_sequence=COLORS,
                     title="Produits par boutique et plateforme")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c6:
        seg_counts = dff["segment_prix"].value_counts().reset_index()
        seg_counts.columns = ["segment","n"]
        if len(seg_counts) > 0:
            fig2 = px.pie(seg_counts, values="n", names="segment",
                          color_discrete_sequence=COLORS,
                          title="Répartition segments de prix")
            fig2.update_layout(**PLOT_LAYOUT)
            fig2.update_traces(textfont_color="#888")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pas de données segment pour ce filtre.")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — TOP-K PRODUCTS
# ══════════════════════════════════════════════════════════════
elif page == "Top-K Products":
    st.markdown('<div class="section-label">Product Rankings / Scoring Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Classement des produits<br>à fort potentiel commercial.</div>', unsafe_allow_html=True)

    k_val = st.slider("Nombre de produits à afficher", 10, 100, 30)

    # ✅ Top-K dynamique sur données filtrées — plus de top_k==1 fixe
    top = (dff
           .sort_values("score_popularite", ascending=False)
           .head(k_val)[["shop","categorie","produit","prix","segment_prix",
                          "score_popularite","cluster_nom","stock"]]
           .reset_index(drop=True))
    top.index += 1

    st.markdown(f'<div class="card"><div class="card-label">Top-K sélectionnés</div><div class="card-value accent">{len(top)}</div><div class="card-sub">sur {len(dff):,} produits filtrés</div></div>', unsafe_allow_html=True)

    if len(top) > 0:
        st.dataframe(
            top.style.background_gradient(subset=["score_popularite"], cmap="YlOrBr"),
            use_container_width=True
        )
    else:
        st.info("Aucun produit pour ce filtre.")

    c1, c2 = st.columns(2)
    with c1:
        topk_shop = (dff.groupby("shop")["score_popularite"]
                     .mean().reset_index()
                     .rename(columns={"score_popularite":"score_moyen"})
                     .sort_values("score_moyen"))
        if len(topk_shop) > 0:
            fig = px.bar(topk_shop, x="score_moyen", y="shop", orientation="h",
                         color="score_moyen",
                         color_continuous_scale=[[0,"#1a1a1a"],[1,ACCENT]],
                         title="Score moyen par boutique",
                         labels={"score_moyen":"Score moyen","shop":""})
            fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        topk_cat = (dff.groupby("categorie")["score_popularite"]
                    .mean().reset_index()
                    .rename(columns={"score_popularite":"score_moyen"})
                    .sort_values("score_moyen").tail(12))
        if len(topk_cat) > 0:
            fig2 = px.bar(topk_cat, x="score_moyen", y="categorie", orientation="h",
                          color="score_moyen",
                          color_continuous_scale=[[0,"#1a1a1a"],[1,"#7e9cc8"]],
                          title="Score moyen par catégorie (Top 12)",
                          labels={"score_moyen":"Score moyen","categorie":""})
            fig2.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="card-label">Détail complet des produits classés</div>', unsafe_allow_html=True)
    detail = (dff.sort_values("score_popularite", ascending=False)
              [["shop","categorie","produit","prix","segment_prix",
                "score_popularite","stock","en_promo"]]
              .reset_index(drop=True))
    detail.index += 1
    detail.index.name = "Rang"
    st.dataframe(
        detail.style.background_gradient(subset=["score_popularite"], cmap="YlOrBr"),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════
# PAGE 3 — PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == "Price Analysis":
    st.markdown('<div class="section-label">Price Intelligence / Market Spread</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Stratégie tarifaire<br>et positionnement des boutiques.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Prix moyen",  f"{dff['prix'].mean():.0f} DH")
    c2.metric("Prix médian", f"{dff['prix'].median():.0f} DH")
    c3.metric("Prix min",    f"{dff['prix'].min():.0f} DH")
    c4.metric("Prix max",    f"{dff['prix'].max():.0f} DH")

    fig_hist = px.histogram(dff, x="prix", nbins=60, color="shop",
                            color_discrete_sequence=COLORS,
                            title="Distribution des prix par boutique",
                            opacity=0.8)
    fig_hist.update_layout(**PLOT_LAYOUT, bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        box_df = dff[dff["prix"] < dff["prix"].quantile(0.95)]
        fig_box = px.box(box_df, x="shop", y="prix", color="shop",
                         color_discrete_sequence=COLORS,
                         title="Dispersion des prix par boutique")
        fig_box.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with c4:
        prix_cat = (dff.groupby("categorie")["prix"].mean()
                    .reset_index().sort_values("prix").tail(15))
        if len(prix_cat) > 0:
            fig_cat = px.bar(prix_cat, x="prix", y="categorie", orientation="h",
                             color="prix",
                             color_continuous_scale=[[0,"#1a1a1a"],[1,ACCENT]],
                             title="Prix moyen Top 15 catégories",
                             labels={"prix":"Prix moyen (DH)","categorie":""})
            fig_cat.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
            st.plotly_chart(fig_cat, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4 — ML CLUSTERING
# ══════════════════════════════════════════════════════════════
elif page == "ML Clustering":
    st.markdown('<div class="section-label">ML Models / KMeans K=5</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Segmentation automatique<br>des produits par profil commercial.</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    c1.metric("Clusters", "5")
    c2.metric("Silhouette Score", "0.490")
    c3.metric("Algorithme", "KMeans")

    st.markdown("""
    <div class="card">
        <div class="card-label">Interprétation des clusters</div>
        <div style="font-size:12px;color:#555;line-height:1.9">
            <span class="tag">Très low-cost</span> Prix &lt; 100 DH — produits d'entrée de gamme<br>
            <span class="tag">Low-cost</span> Prix 100–200 DH — segment accessible, Kiabi dominant<br>
            <span class="tag">Mid-range</span> Prix 200–400 DH — segment principal, BeautyMarket + Justyol<br>
            <span class="tag">Premium</span> Prix 400–800 DH — segment haut de gamme, Lasolda + Lhmiza<br>
            <span class="tag">Luxe</span> Prix &gt; 800 DH — électroménager/tech premium
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "cluster_nom" in dff.columns and dff["cluster_nom"].notna().any():
        fig_sc = px.scatter(dff, x="log_prix", y="score_popularite",
                            color="cluster_nom",
                            hover_data=["produit","prix","shop"],
                            color_discrete_sequence=COLORS,
                            title="Clusters KMeans — Log Prix vs Score Popularité")
        fig_sc.update_traces(marker=dict(size=5, opacity=0.55))
        fig_sc.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_sc, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            profil = dff.groupby("cluster_nom").agg(
                prix_moyen=("prix","mean"),
                score_moyen=("score_popularite","mean"),
                promo_pct=("promo_bin","mean"),
                nb=("produit","count")
            ).reset_index().round(2)
            st.dataframe(
                profil.style.background_gradient(subset=["prix_moyen"], cmap="YlOrBr"),
                use_container_width=True
            )
        with c6:
            cluster_shop = dff.groupby(["cluster_nom","shop"]).size().reset_index(name="n")
            fig_cl = px.bar(cluster_shop, x="cluster_nom", y="n", color="shop",
                            color_discrete_sequence=COLORS,
                            title="Répartition clusters par boutique",
                            barmode="stack")
            fig_cl.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig_cl, use_container_width=True)
    else:
        st.info("Données de clustering non disponibles pour ce filtre.")

# ══════════════════════════════════════════════════════════════
# PAGE 5 — RANDOM FOREST
# ══════════════════════════════════════════════════════════════
elif page == "Random Forest":
    st.markdown('<div class="section-label">ML Models / Classification supervisée</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prédiction Top-K<br>par Random Forest.</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Algorithme", "Random Forest")
    c2.metric("Arbres", "200")
    c3.metric("F1 Score (CV)", "0.760")
    c4.metric("Validation", "5-fold CV")

    st.markdown("""
    <div class="card">
        <div class="card-label">Variables utilisées pour la prédiction</div>
        <div style="margin-top:8px">
            <span class="tag">log_prix</span>
            <span class="tag">promo_bin</span>
            <span class="tag">stock_bin</span>
            <span class="tag">shop_id</span>
            <span class="tag">categorie_id</span>
            <span class="tag">platform_id</span>
            <span class="tag">cluster</span>
        </div>
        <div style="font-size:11px;color:#444;margin-top:12px">
            score_popularite exclu pour éviter la fuite de données (data leakage)
        </div>
    </div>
    """, unsafe_allow_html=True)

    import os
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png",
                 caption="Matrice de confusion — Random Forest", width=700)
    else:
        st.info("confusion_matrix.png non trouvé — lancez ml_pipeline.py d'abord.")

    if os.path.exists("feature_importance.png"):
        st.image("feature_importance.png",
                 caption="Importance des variables", width=700)
    else:
        st.info("feature_importance.png non trouvé — lancez ml_pipeline.py d'abord.")

# ══════════════════════════════════════════════════════════════
# PAGE 6 — APRIORI RULES
# ══════════════════════════════════════════════════════════════
elif page == "Apriori Rules":
    st.markdown('<div class="section-label">Association Rules / Apriori Algorithm</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Règles d\'association<br>entre catégories de produits.</div>', unsafe_allow_html=True)

    if len(rules) > 0:
        c1,c2,c3 = st.columns(3)
        c1.metric("Règles trouvées", f"{len(rules):,}")
        c2.metric("Lift moyen",      f"{rules['lift'].mean():.2f}")
        c3.metric("Confiance moy.",  f"{rules['confidence'].mean():.2f}")

        st.markdown("""
        <div class="card">
            <div class="card-label">Lecture des métriques</div>
            <div style="font-size:11px;color:#555;line-height:1.9">
                <span class="accent">Support</span> = fréquence d'apparition de la règle dans le dataset<br>
                <span class="accent">Confidence</span> = probabilité que B soit présent si A est présent<br>
                <span class="accent">Lift</span> = force de l'association (lift=5 = très forte)
            </div>
        </div>
        """, unsafe_allow_html=True)

        rules["antecedents"] = rules["antecedents"].astype(str)
        rules["consequents"] = rules["consequents"].astype(str)

        fig_rules = px.scatter(
            rules.head(100), x="support", y="confidence",
            size="lift", color="lift",
            color_continuous_scale=[[0,"#1a1a1a"],[1,ACCENT]],
            hover_data=["antecedents","consequents"],
            title="Support vs Confiance (taille = lift)"
        )
        fig_rules.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig_rules, use_container_width=True)

        st.dataframe(
            rules[["antecedents","consequents","support","confidence","lift"]]
            .head(20).reset_index(drop=True),
            use_container_width=True
        )
    else:
        st.warning("regles_association.csv non trouvé.")

# ══════════════════════════════════════════════════════════════
# PAGE 7 — LLM INSIGHTS
# ══════════════════════════════════════════════════════════════
elif page == "LLM Insights":
    st.markdown('<div class="section-label">LLM / Enrichissement automatique</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Synthèse intelligente<br>par modèle de langage.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-label">Architecture LLM</div>
        <div style="font-size:12px;color:#555;line-height:1.9">
            Ce module utilise l'API Claude pour générer automatiquement des synthèses business
            à partir des résultats ML. Il suit le
            <span class="accent">Model Context Protocol (MCP)</span> d'Anthropic
            pour une interaction responsable et contrôlée avec les outils.
        </div>
    </div>
    """, unsafe_allow_html=True)

    top5 = (dff.sort_values("score_popularite", ascending=False)
            .head(5)[["shop","produit","prix","categorie"]]
            .to_string(index=False))

    cluster_summary = (dff.groupby("cluster_nom")["prix"].mean().round(0).to_dict()
                       if "cluster_nom" in dff.columns else {})

    nb_rules = len(rules)
    lift_moy = f"{rules['lift'].mean():.2f}" if nb_rules > 0 else "N/A"

    contexte = f"""
Dataset : {len(dff):,} produits | {dff['shop'].nunique()} boutiques | {dff['categorie'].nunique()} catégories
Prix moyen : {dff['prix'].mean():.0f} DH | Médian : {dff['prix'].median():.0f} DH
En promotion : {int(dff['promo_bin'].sum()):,} ({dff['promo_bin'].mean()*100:.1f}%)
Clusters KMeans : {cluster_summary}
Silhouette Score : 0.490 | F1 Random Forest : 0.760
Règles Apriori : {nb_rules:,} règles | Lift moyen : {lift_moy}
Top 5 produits :
{top5}
    """

    st.markdown('<div class="card-label">Contexte transmis au LLM</div>', unsafe_allow_html=True)
    st.code(contexte, language="text")

    api_key = st.text_input("Clé API Anthropic", type="password",
                             placeholder="sk-ant-...")
    prompt_choice = st.selectbox("Type d'analyse", [
        "Synthèse générale du marché",
        "Recommandations stratégiques pour les boutiques",
        "Analyse des tendances prix et promotions",
        "Interprétation des clusters KMeans",
        "Opportunités détectées par les règles Apriori"
    ])

    if st.button("Générer la synthèse LLM", type="primary"):
        if not api_key:
            st.error("Entrez votre clé API Anthropic")
        else:
            import anthropic
            prompts = {
                "Synthèse générale du marché":
                    f"Tu es un analyste e-commerce expert. Voici les données :\n{contexte}\n\nGénère une synthèse business claire en français (5-7 phrases).",
                "Recommandations stratégiques pour les boutiques":
                    f"Données :\n{contexte}\n\nPropose 5 recommandations stratégiques concrètes.",
                "Analyse des tendances prix et promotions":
                    f"Données :\n{contexte}\n\nAnalyse la stratégie de prix et les promotions détectées.",
                "Interprétation des clusters KMeans":
                    f"Données :\n{contexte}\n\nInterprète les clusters KMeans d'un point de vue business.",
                "Opportunités détectées par les règles Apriori":
                    f"Données :\n{contexte}\n\nQuelles opportunités de cross-selling les règles d'association révèlent-elles ?"
            }

            with st.spinner("Génération en cours..."):
                try:
                    client = anthropic.Anthropic(api_key=api_key)
                    message = client.messages.create(
                        model="claude-opus-4-6",
                        max_tokens=800,
                        messages=[{"role":"user","content": prompts[prompt_choice]}]
                    )
                    reponse = message.content[0].text
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-label">Synthèse générée par Claude</div>
                        <div style="font-size:13px;color:#888;line-height:1.9;margin-top:8px">
                            {reponse.replace(chr(10),'<br>')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erreur API : {e}")
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-label">Synthèse exemple (sans LLM)</div>
                        <div style="font-size:12px;color:#666;line-height:1.9">
                            Le marché e-commerce marocain analysé présente {len(dff):,} produits
                            avec un prix médian de {dff['prix'].median():.0f} DH, dominé par le
                            segment mid-range. Justyol représente la plus grande part du catalogue
                            avec {dff[dff['shop']=='Justyol'].shape[0]:,} produits.
                            Le clustering KMeans révèle 5 segments distincts (Silhouette=0.49).
                            Les règles d'association montrent un lift de 5.0, suggérant de fortes
                            opportunités de cross-selling entre Mode et Beauté.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)