import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── CONFIG ────────────────────────────────────────────────────────────────────

DB_PATH = "put your path"

FEATURES = [
    'tempo', 'rms_energy', 'spectral_centroid_mean',
    'spectral_rolloff_mean', 'zero_crossing_rate_mean', 'chroma_mean',
    'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean'
]

FEATURE_LABELS = {
    'tempo': 'Tempo (BPM)',
    'rms_energy': 'Énergie RMS',
    'spectral_centroid_mean': 'Centroïde spectral',
    'spectral_rolloff_mean': 'Rolloff spectral',
    'zero_crossing_rate_mean': 'Zero Crossing Rate',
    'chroma_mean': 'Chroma (Harmonie)',
    'mfcc_1_mean': 'MFCC 1 (Timbre)',
    'mfcc_2_mean': 'MFCC 2',
    'mfcc_3_mean': 'MFCC 3',
    'mfcc_4_mean': 'MFCC 4',
    'mfcc_5_mean': 'MFCC 5',
}

# 5 features significatives après analyse de robustesse
SIG_FEATURES = [
    'rms_energy', 'spectral_centroid_mean', 'spectral_rolloff_mean',
    'zero_crossing_rate_mean', 'chroma_mean'
]
SIG_LABELS = [
    'Énergie RMS', 'Centroïde spectral', 'Rolloff spectral',
    'Zero Crossing Rate', 'Chroma (Harmonie)'
]

# Palette Deezer
C_EDITORIAL = "#FF0092"
C_HUMAN     = "#00C7F2"
C_ACCENT    = "#FFED00"
BG          = "#0A0A0A"
SURFACE     = "#141414"
TEXT        = "#F0F0F0"
SUBTEXT     = "#666666"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   SURFACE,
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTEXT,
    "ytick.color":      SUBTEXT,
    "text.color":       TEXT,
    "grid.color":       "#2A2A2A",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
})

# ─── DATA ──────────────────────────────────────────────────────────────────────

con = sqlite3.connect(DB_PATH)
df = pd.read_sql("""
    SELECT
        p.playlist_id, p.name, p.type, p.fans, p.tracks_total,
        af.tempo, af.rms_energy, af.spectral_centroid_mean,
        af.spectral_rolloff_mean, af.zero_crossing_rate_mean,
        af.chroma_mean, af.mfcc_1_mean, af.mfcc_2_mean,
        af.mfcc_3_mean, af.mfcc_4_mean, af.mfcc_5_mean
    FROM playlists p
    JOIN playlist_tracks pt ON p.playlist_id = pt.playlist_id
    JOIN tracks t ON pt.track_id = t.track_id
    JOIN audio_features af ON t.track_id = af.track_id
""", con)
con.close()

variance_df = df.groupby(['playlist_id', 'name', 'type'])[FEATURES].std().reset_index()

editorial = variance_df[variance_df['type'] == 'editorial']
human     = variance_df[variance_df['type'] == 'human']

# p-values Mann-Whitney
pvalues = {}
for f in FEATURES:
    _, p = stats.mannwhitneyu(
        editorial[f].dropna(),
        human[f].dropna(),
        alternative='less'
    )
    pvalues[f] = p


# ─── FIGURE 1 : BOXPLOTS ──────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle(
    "Variance acoustique intra-playlist\nÉditoriales Deezer vs Playlists humaines",
    fontsize=14, fontweight='bold', color=TEXT, y=0.98, linespacing=1.6
)
axes_flat = axes.flatten()

for i, feature in enumerate(FEATURES):
    ax = axes_flat[i]
    ed_vals = editorial[feature].dropna()
    hu_vals = human[feature].dropna()

    bp = ax.boxplot(
        [ed_vals, hu_vals],
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color=SUBTEXT, linewidth=1),
        capprops=dict(color=SUBTEXT, linewidth=1),
        flierprops=dict(marker='o', markerfacecolor=SUBTEXT,
                        markeredgecolor='none', markersize=3, alpha=0.5),
    )
    bp['boxes'][0].set_facecolor(C_EDITORIAL)
    bp['boxes'][0].set_alpha(0.85)
    bp['boxes'][1].set_facecolor(C_HUMAN)
    bp['boxes'][1].set_alpha(0.85)

    p = pvalues[feature]
    sig_label = "p={:.4f}{}".format(p, " ***" if p < 0.01 else " *" if p < 0.05 else "")
    color_sig = C_EDITORIAL if p < 0.05 else SUBTEXT

    ax.set_title(FEATURE_LABELS[feature], fontsize=9, color=TEXT, pad=6)
    ax.text(0.97, 0.97, sig_label, transform=ax.transAxes,
            fontsize=7.5, color=color_sig, ha='right', va='top')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Éditoriale", "Humaine"], fontsize=8)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

axes_flat[-1].set_visible(False)

legend_patches = [
    mpatches.Patch(color=C_EDITORIAL, alpha=0.85, label="Éditoriale Deezer"),
    mpatches.Patch(color=C_HUMAN,     alpha=0.85, label="Humaine populaire"),
]
fig.legend(handles=legend_patches, loc='lower right',
           framealpha=0, fontsize=9, bbox_to_anchor=(0.97, 0.04))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("boxplots_variance.png", dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print("Sauvegardé : boxplots_variance.png")


# ─── FIGURE 2 : VIOLIN PLOTS (5 features robustes) ────────────────────────────

fig, axes = plt.subplots(1, 5, figsize=(20, 6))
fig.suptitle(
    "Distribution des variances — 5 features significatives après contrôle de taille (p < 0.05)",
    fontsize=12, fontweight='bold', color=TEXT, y=1.01
)

for ax, feature, label in zip(axes, SIG_FEATURES, SIG_LABELS):
    ed_vals = editorial[feature].dropna().values
    hu_vals = human[feature].dropna().values

    parts_ed = ax.violinplot([ed_vals], positions=[1],
                             showmedians=True, showextrema=False)
    parts_hu = ax.violinplot([hu_vals], positions=[2],
                             showmedians=True, showextrema=False)

    for pc in parts_ed['bodies']:
        pc.set_facecolor(C_EDITORIAL)
        pc.set_alpha(0.75)
    parts_ed['cmedians'].set_color("white")
    parts_ed['cmedians'].set_linewidth(2)

    for pc in parts_hu['bodies']:
        pc.set_facecolor(C_HUMAN)
        pc.set_alpha(0.75)
    parts_hu['cmedians'].set_color("white")
    parts_hu['cmedians'].set_linewidth(2)

    p = pvalues[feature]
    ax.set_title(f"{label}\np = {p:.4f}", fontsize=9, color=TEXT, linespacing=1.8)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Éditoriale", "Humaine"], fontsize=8)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("violins_significatifs.png", dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print("Sauvegardé : violins_significatifs.png")


# ─── FIGURE 3 : RÉSUMÉ P-VALUES ───────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(
    "Significativité statistique par feature acoustique\n(test Mann-Whitney U, hypothèse : éditoriale < humaine)",
    fontsize=12, fontweight='bold', color=TEXT, y=1.02, linespacing=1.7
)

labels_fig  = [FEATURE_LABELS[f] for f in FEATURES]
pvals       = [pvalues[f] for f in FEATURES]
colors      = [C_EDITORIAL if p < 0.05 else "#444444" for p in pvals]

bars = ax.barh(labels_fig, [-np.log10(p) for p in pvals],
               color=colors, height=0.6, alpha=0.9)

ax.axvline(x=-np.log10(0.05), color=TEXT, linewidth=1,
           linestyle='--', alpha=0.5, label='seuil p=0.05')

for bar, p in zip(bars, pvals):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f"p={p:.4f}", va='center', fontsize=8,
            color=C_EDITORIAL if p < 0.05 else SUBTEXT)

ax.set_xlabel("−log₁₀(p)", color=TEXT, fontsize=10)
ax.legend(framealpha=0, fontsize=9)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("pvalues_summary.png", dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print("Sauvegardé : pvalues_summary.png")


# ─── FIGURE 4 : RADAR CHART ───────────────────────────────────────────────────

radar_features = [
    'tempo', 'rms_energy', 'spectral_centroid_mean',
    'spectral_rolloff_mean', 'zero_crossing_rate_mean',
    'chroma_mean', 'mfcc_2_mean'
]
radar_labels = ['Tempo', 'Énergie', 'Brillance', 'Rolloff', 'Texture', 'Harmonie', 'Timbre']

df_radar = variance_df[radar_features].copy()
df_radar_norm = (df_radar - df_radar.min()) / (df_radar.max() - df_radar.min())
df_radar_norm['type'] = variance_df['type'].values

ed_means = df_radar_norm[df_radar_norm['type'] == 'editorial'][radar_features].mean().values
hu_means = df_radar_norm[df_radar_norm['type'] == 'human'][radar_features].mean().values

N = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
ed_means = np.append(ed_means, ed_means[0])
hu_means = np.append(hu_means, hu_means[0])

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor(BG)
ax.set_facecolor(SURFACE)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, color=TEXT, fontsize=10)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75])
ax.set_yticklabels(["0.25", "0.50", "0.75"], color=SUBTEXT, fontsize=7)
ax.spines['polar'].set_color("#2A2A2A")
ax.grid(color="#2A2A2A", linewidth=0.8)

ax.plot(angles, ed_means, color=C_EDITORIAL, linewidth=2.5, linestyle='solid')
ax.fill(angles, ed_means, color=C_EDITORIAL, alpha=0.20)
ax.plot(angles, hu_means, color=C_HUMAN, linewidth=2.5, linestyle='solid')
ax.fill(angles, hu_means, color=C_HUMAN, alpha=0.20)

legend_patches = [
    mpatches.Patch(color=C_EDITORIAL, alpha=0.8, label="Éditoriale Deezer"),
    mpatches.Patch(color=C_HUMAN,     alpha=0.8, label="Humaine populaire"),
]
ax.legend(handles=legend_patches, loc='upper right',
          bbox_to_anchor=(1.35, 1.15), framealpha=0, fontsize=10)

ax.set_title(
    "Profil de variance acoustique moyen\n(normalisé — plus c'est grand, plus c'est variable)",
    color=TEXT, fontsize=12, fontweight='bold', pad=25, linespacing=1.7
)

plt.tight_layout()
plt.savefig("radar_profil.png", dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print("Sauvegardé : radar_profil.png")


# ─── FIGURE 5 : PCA SCATTER ───────────────────────────────────────────────────

variance_pca = variance_df.dropna()
X = variance_pca[FEATURES].values
labels_pca = variance_pca['type'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(11, 8))

for ptype, color, label in [('editorial', C_EDITORIAL, 'Éditoriale Deezer'),
                              ('human', C_HUMAN, 'Humaine populaire')]:
    mask = labels_pca == ptype
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=color, label=label,
        s=70, alpha=0.85, edgecolors='none', zorder=3
    )

# Ellipses de confiance (1 écart-type)
for ptype, color in [('editorial', C_EDITORIAL), ('human', C_HUMAN)]:
    mask = labels_pca == ptype
    pts = X_pca[mask]
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        edgecolor=color, fc=color, alpha=0.08, lw=1.5, linestyle='--'
    )
    ax.add_patch(ellipse)

ax.set_xlabel(f"PC1 — {var_explained[0]:.1f}% de variance expliquée", fontsize=10)
ax.set_ylabel(f"PC2 — {var_explained[1]:.1f}% de variance expliquée", fontsize=10)
ax.set_title(
    f"PCA des variances acoustiques intra-playlist\n"
    f"Variance totale expliquée : {sum(var_explained):.1f}%",
    fontsize=13, fontweight='bold', color=TEXT, linespacing=1.7
)
ax.legend(framealpha=0, fontsize=10)
ax.grid(True)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("pca_scatter.png", dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print("Sauvegardé : pca_scatter.png")


# ─── FIGURE 6 : PCA LOADINGS ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

loadings = pca.components_.T
feature_labels_short = [f.replace('_mean', '').replace('_', ' ') for f in FEATURES]

x = np.arange(len(FEATURES))
width = 0.35

ax.bar(x - width/2, loadings[:, 0], width,
       color=C_EDITORIAL, alpha=0.85, label=f'PC1 ({var_explained[0]:.1f}%)')
ax.bar(x + width/2, loadings[:, 1], width,
       color=C_HUMAN, alpha=0.85, label=f'PC2 ({var_explained[1]:.1f}%)')

ax.axhline(0, color=SUBTEXT, linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(feature_labels_short, rotation=35, ha='right', fontsize=8)
ax.set_ylabel("Loading (contribution)", fontsize=10)
ax.set_title(
    "Contribution des features acoustiques aux axes principaux (PCA)",
    fontsize=12, fontweight='bold', color=TEXT
)
ax.legend(framealpha=0, fontsize=9)
ax.grid(axis='y')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("pca_loadings.png", dpi=150, bbox_inches='tight',
            facecolor=BG, edgecolor='none')
plt.show()
print("Sauvegardé : pca_loadings.png")