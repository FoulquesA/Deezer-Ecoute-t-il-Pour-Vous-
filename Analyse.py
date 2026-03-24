import pandas
import numpy as np
import pandas as pd
from scipy import stats

# loading in modules
import sqlite3

# creating file path
dbfile = "put the path"
# Create a SQL connection to our SQLite database
con = sqlite3.connect(dbfile)

# creating cursor
cur = con.cursor()

# reading all table names
table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]

print(table_list)

 #─── Data cleaning ───────────────────────────────────────────────────
#cur.execute("DELETE FROM tracks WHERE duration > 600")
#cur.execute("DELETE FROM audio_features WHERE tempo = 0")
#cur.execute("DELETE FROM audio_features WHERE mfcc_1_mean < -500")
#con.commit()



playlists = pd.read_sql("SELECT * FROM playlists", con)
tracks = pd.read_sql("SELECT * FROM tracks", con)
audio_features = pd.read_sql("SELECT * FROM audio_features", con)
playlist_tracks = pd.read_sql("SELECT * FROM playlist_tracks", con)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#print(playlists.describe( ))
#print(tracks.describe( ))
#print(playlist_tracks.describe( ))
#print(audio_features.describe( ))




df = pd.read_sql("""
    SELECT 
        p.playlist_id, p.name, p.type,
        p.fans, p.tracks_total,
        af.tempo, af.rms_energy,
        af.spectral_centroid_mean, af.spectral_rolloff_mean,
        af.zero_crossing_rate_mean, af.chroma_mean,
        af.mfcc_1_mean, af.mfcc_2_mean, af.mfcc_3_mean,
        af.mfcc_4_mean, af.mfcc_5_mean
    FROM playlists p
    JOIN playlist_tracks pt ON p.playlist_id = pt.playlist_id
    JOIN tracks t ON pt.track_id = t.track_id
    JOIN audio_features af ON t.track_id = af.track_id
""", con)

features = ['tempo', 'rms_energy', 'spectral_centroid_mean',
            'spectral_rolloff_mean', 'zero_crossing_rate_mean',
            'chroma_mean', 'mfcc_1_mean', 'mfcc_2_mean',
            'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean']

variance_df = df.groupby(['playlist_id', 'name', 'type'])[features].std().reset_index()
#print(variance_df)


 #─── Test hypothèse : les éditoriales ont une variance inférieure aux humaines ───────────────────────────────────────────────────
editorial = variance_df[variance_df['type'] == 'editorial']
human = variance_df[variance_df['type'] == 'human']

for feature in features:
    stat, p = stats.mannwhitneyu(
        editorial[feature].dropna(),
        human[feature].dropna(),
        alternative='less'  
    )
    #print(f"{feature}: p={p:.4f} {'*** SIGNIFICATIF' if p < 0.05 else ''}")




 #─── Test de robustesse ───────────────────────────────────────────────────


FEATURES = [
    'tempo', 'rms_energy', 'spectral_centroid_mean',
    'spectral_rolloff_mean', 'zero_crossing_rate_mean', 'chroma_mean',
    'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean', 'mfcc_5_mean'
]



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

# Variance intra-playlist + taille réelle (nb tracks avec features)
variance_df = df.groupby(['playlist_id', 'name', 'type', 'tracks_total'])[FEATURES].std().reset_index()

# Taille réelle = nb tracks effectivement analysés par playlist
track_counts = df.groupby('playlist_id')[FEATURES[0]].count().reset_index()
track_counts.columns = ['playlist_id', 'n_tracks_analysed']
variance_df = variance_df.merge(track_counts, on='playlist_id')

# Variable binaire : 1 = editorial, 0 = human
variance_df['is_editorial'] = (variance_df['type'] == 'editorial').astype(int)

print("=" * 65)
print("ANALYSE DE ROBUSTESSE — Contrôle par taille de playlist")
print("=" * 65)
print(f"\nPlaylists éditoriales : {variance_df['is_editorial'].sum()}")
print(f"Playlists humaines    : {(variance_df['is_editorial'] == 0).sum()}")
print(f"Taille médiane éditoriales : {variance_df[variance_df['is_editorial']==1]['n_tracks_analysed'].median():.0f} tracks")
print(f"Taille médiane humaines    : {variance_df[variance_df['is_editorial']==0]['n_tracks_analysed'].median():.0f} tracks")

print("\n" + "-" * 65)
print(f"{'Feature':<30} {'β editorial':>12} {'β taille':>10} {'p (editorial)':>14}")
print("-" * 65)

from numpy.linalg import lstsq

results = []

for feature in FEATURES:
    y = variance_df[feature].dropna()
    idx = y.index

    X = np.column_stack([
        np.ones(len(idx)),
        variance_df.loc[idx, 'is_editorial'].values,
        variance_df.loc[idx, 'n_tracks_analysed'].values,
    ])

    # Régression OLS manuelle
    coeffs, _, _, _ = lstsq(X, y.values, rcond=None)
    intercept, beta_editorial, beta_size = coeffs

    # Résidus et t-test sur beta_editorial
    y_pred = X @ coeffs
    residuals = y.values - y_pred
    n = len(y)
    k = 3
    sigma2 = np.sum(residuals**2) / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_editorial = np.sqrt(cov[1, 1])
    t_stat = beta_editorial / se_editorial
    p_val = stats.t.sf(np.abs(t_stat), df=n-k) * 2  # two-tailed

    # Correction pour hypothèse directionnelle (éditoriale < humaine)
    p_one_tail = p_val / 2 if beta_editorial < 0 else 1 - p_val / 2

    sig = "***" if p_one_tail < 0.01 else "*" if p_one_tail < 0.05 else ""
    print(f"{feature:<30} {beta_editorial:>12.6f} {beta_size:>10.6f} {p_one_tail:>12.4f}  {sig}")

    results.append({
        'feature': feature,
        'beta_editorial': beta_editorial,
        'beta_size': beta_size,
        'p_one_tail': p_one_tail,
        'significant': p_one_tail < 0.05
    })

print("-" * 65)
print("\n* p < 0.05   *** p < 0.01")
print("β editorial négatif = les éditoriales sont moins variables, toutes choses égales par ailleurs")

sig_robust = [r['feature'] for r in results if r['significant'] and r['beta_editorial'] < 0]
print(f"\nFeatures significatives après contrôle de la taille : {len(sig_robust)}")
for f in sig_robust:
    print(f"  -> {f}")


con.close()
