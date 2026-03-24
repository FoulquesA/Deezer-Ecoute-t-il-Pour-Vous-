import sqlite3

def create_db():
    with sqlite3.connect("deezer_study.db") as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS playlists (
                playlist_id INTEGER PRIMARY KEY,
                name TEXT,
                owner_id TEXT,
                owner_name TEXT,
                fans INTEGER,
                tracks_total INTEGER,
                type TEXT CHECK(type IN ('editorial', 'human'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id INTEGER PRIMARY KEY,
                title TEXT,
                artist_id INTEGER,
                artist_name TEXT,
                duration INTEGER,
                rank INTEGER,
                explicit INTEGER,
                preview_url TEXT,
                deezer_bpm REAL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS playlist_tracks (
                playlist_id INTEGER,
                track_id INTEGER,
                position INTEGER,
                PRIMARY KEY (playlist_id, track_id),
                FOREIGN KEY (playlist_id) REFERENCES playlists(playlist_id),
                FOREIGN KEY (track_id) REFERENCES tracks(track_id)
            )
        """)

        # Features extraites par librosa depuis les previews MP3
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audio_features (
                track_id INTEGER PRIMARY KEY,
                tempo REAL,
                rms_energy REAL,
                spectral_centroid_mean REAL,
                spectral_rolloff_mean REAL,
                zero_crossing_rate_mean REAL,
                chroma_mean REAL,
                mfcc_1_mean REAL,
                mfcc_2_mean REAL,
                mfcc_3_mean REAL,
                mfcc_4_mean REAL,
                mfcc_5_mean REAL,
                FOREIGN KEY (track_id) REFERENCES tracks(track_id)
            )
        """)

        conn.commit()
        print("Base de données créée avec succès : deezer_study.db")

create_db()
