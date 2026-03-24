import sqlite3
import requests
import librosa
import numpy as np
import tempfile
import os
import time

DB_PATH = "deezer_study.db"


def get_tracks_without_features(conn):
    """Récupère les tracks qui ont une preview_url mais pas encore de features."""
    cursor = conn.execute("""
        SELECT t.track_id, t.preview_url
        FROM tracks t
        LEFT JOIN audio_features af ON t.track_id = af.track_id
        WHERE t.preview_url != ''
          AND t.preview_url IS NOT NULL
          AND af.track_id IS NULL
    """)
    return cursor.fetchall()


def get_fresh_preview_url(track_id):
    """Récupère une URL de preview fraîche depuis l'API Deezer."""
    try:
        r = requests.get(f"https://api.deezer.com/track/{track_id}", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "error" in data:
            return None
        return data.get("preview")
    except Exception:
        return None


def download_preview(url):
    """Télécharge le MP3 dans un fichier temporaire, retourne le chemin."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.deezer.com/"
    }
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def extract_features(filepath):
    """
    Extrait les features acoustiques depuis un fichier audio.
    Retourne un dict ou None en cas d'échec.
    """
    y, sr = librosa.load(filepath, sr=22050, mono=True)

    if len(y) < sr:  # moins d'une seconde, inutilisable
        return None

    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)

    # Energie RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))

    # Centroïde spectral (brightness)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_centroid_mean = float(np.mean(spec_centroid))

    # Spectral rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spec_rolloff_mean = float(np.mean(spec_rolloff))

    # Zero crossing rate (texture)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = float(np.mean(zcr))

    # Chroma (tonalité)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = float(np.mean(chroma))

    # MFCCs (timbre) — on garde les 5 premiers coefficients
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    mfcc_means = [float(np.mean(mfccs[i])) for i in range(5)]

    return {
        "tempo": tempo,
        "rms_energy": rms_mean,
        "spectral_centroid_mean": spec_centroid_mean,
        "spectral_rolloff_mean": spec_rolloff_mean,
        "zero_crossing_rate_mean": zcr_mean,
        "chroma_mean": chroma_mean,
        "mfcc_1_mean": mfcc_means[0],
        "mfcc_2_mean": mfcc_means[1],
        "mfcc_3_mean": mfcc_means[2],
        "mfcc_4_mean": mfcc_means[3],
        "mfcc_5_mean": mfcc_means[4],
    }


def insert_features(conn, track_id, features):
    conn.execute("""
        INSERT OR IGNORE INTO audio_features (
            track_id, tempo, rms_energy,
            spectral_centroid_mean, spectral_rolloff_mean,
            zero_crossing_rate_mean, chroma_mean,
            mfcc_1_mean, mfcc_2_mean, mfcc_3_mean, mfcc_4_mean, mfcc_5_mean
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        track_id,
        features["tempo"], features["rms_energy"],
        features["spectral_centroid_mean"], features["spectral_rolloff_mean"],
        features["zero_crossing_rate_mean"], features["chroma_mean"],
        features["mfcc_1_mean"], features["mfcc_2_mean"], features["mfcc_3_mean"],
        features["mfcc_4_mean"], features["mfcc_5_mean"],
    ))


def extract_all():
    with sqlite3.connect(DB_PATH) as conn:
        tracks = get_tracks_without_features(conn)
        total = len(tracks)
        print(f"{total} tracks à traiter.\n")

        success = 0
        errors = 0

        for i, (track_id, preview_url) in enumerate(tracks):
            print(f"[{i+1}/{total}] Track {track_id}...", end=" ")

            tmp_path = None
            try:
                fresh_url = get_fresh_preview_url(track_id)
                if not fresh_url:
                    print("pas d'URL preview")
                    errors += 1
                    continue

                tmp_path = download_preview(fresh_url)
                if not tmp_path:
                    print("preview inaccessible")
                    errors += 1
                    continue

                features = extract_features(tmp_path)
                if not features:
                    print("audio trop court")
                    errors += 1
                    continue

                insert_features(conn, track_id, features)

                # Commit par batch de 50
                if (i + 1) % 50 == 0:
                    conn.commit()

                print(f"OK (tempo={features['tempo']:.1f} BPM)")
                success += 1
                time.sleep(0.05)

            except Exception as e:
                print(f"Erreur : {e}")
                errors += 1

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        conn.commit()
        print(f"\nExtraction terminée : {success} OK, {errors} erreurs sur {total} tracks.")


extract_all()