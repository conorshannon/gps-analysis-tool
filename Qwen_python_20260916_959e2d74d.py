"""
Sport GPS Analysis Tool — Single-file edition
==============================================
Fully functional GPS analysis platform: CSV import, SQLite storage,
individual & team insights, ACWR, forecasting, and reports.

Run locally:  streamlit run app.py
Deploy:       Push to GitHub → deploy on share.streamlit.io
"""

import os
import io
import json
import hashlib
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =============================================================================
# SECTION 1: CONSTANTS & CONFIG
# =============================================================================

DB_PATH = os.environ.get("SPORT_GPS_DB", "sport_gps.db")

DEFAULT_ZONES = {
    "zone1_walk":   (0.0, 1.9),
    "zone2_jog":    (2.0, 3.4),
    "zone3_run":    (3.5, 5.4),
    "zone4_hsr":    (5.5, 6.9),
    "zone5_sprint": (7.0, np.inf),
}

HSR_THRESHOLD = 5.5
SPRINT_THRESHOLD = 7.0
ACC_THRESHOLD = 2.5
DEC_THRESHOLD = -2.5

COLUMN_ALIASES = {
    "timestamp":   ["timestamp", "time", "datetime", "date_time", "fecha", "hora"],
    "player_id":   ["player_id", "playerid", "id", "player", "jugador_id"],
    "player_name": ["player_name", "name", "athlete", "jugador"],
    "team":        ["team", "squad", "equipo"],
    "position":    ["position", "role", "pos"],
    "speed":       ["speed", "velocity", "velocidad", "v", "spd"],
    "distance":    ["distance", "dist", "distancia"],
    "acc_x":       ["acc_x", "ax", "accel_x", "acx"],
    "acc_y":       ["acc_y", "ay", "accel_y", "acy"],
    "acc_z":       ["acc_z", "az", "accel_z", "acz"],
    "heart_rate":  ["heart_rate", "hr", "bpm", "frecuencia_cardiaca"],
}


# =============================================================================
# SECTION 2: DATA MODELS
# =============================================================================

@dataclass
class Player:
    player_id: str
    name: str
    team: str = "Senior"
    position: str = "Unknown"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Session:
    session_id: str
    date: str
    team: str
    session_type: str
    duration_s: float
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PlayerSessionMetrics:
    session_id: str
    player_id: str
    total_distance_m: float
    relative_distance_m_min: float
    hsr_distance_m: float
    sprint_distance_m: float
    max_speed_m_s: float
    mean_speed_m_s: float
    player_load: float
    accelerations: int
    decelerations: int
    high_intensity_distance_m: float
    duration_s: float
    speed_zones: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_row(cls, row: dict) -> "PlayerSessionMetrics":
        d = dict(row)
        d["speed_zones"] = json.loads(d.get("speed_zones", "{}") or "{}")
        d["metadata"] = json.loads(d.get("metadata", "{}") or "{}")
        return cls(**d)


@dataclass
class ACWRResult:
    player_id: str
    date: str
    acute_load: float
    chronic_load_21: float
    chronic_load_28: float
    chronic_load_35: float
    ratio_7_21: float
    ratio_7_28: float
    ratio_7_35: float
    risk_band: str


@dataclass
class Forecast:
    player_id: str
    target_date: str
    predicted_player_load: float
    predicted_distance_m: float
    predicted_hsr_m: float
    confidence_low: float
    confidence_high: float
    method: str


# =============================================================================
# SECTION 3: DATABASE
# =============================================================================

class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY, name TEXT, team TEXT,
                    position TEXT, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, date TEXT, team TEXT,
                    session_type TEXT, duration_s REAL, notes TEXT, created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS player_session_metrics (
                    session_id TEXT, player_id TEXT,
                    total_distance_m REAL, relative_distance_m_min REAL,
                    hsr_distance_m REAL, sprint_distance_m REAL,
                    max_speed_m_s REAL, mean_speed_m_s REAL,
                    player_load REAL, accelerations INTEGER, decelerations INTEGER,
                    high_intensity_distance_m REAL, duration_s REAL,
                    speed_zones TEXT, metadata TEXT,
                    PRIMARY KEY (session_id, player_id)
                );
                CREATE INDEX IF NOT EXISTS idx_psm_session
                    ON player_session_metrics(session_id);
                CREATE INDEX IF NOT EXISTS idx_psm_player
                    ON player_session_metrics(player_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);
            """)

    def upsert_player(self, p: Player):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO players (player_id, name, team, position, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET
                     name=excluded.name, team=excluded.team, position=excluded.position""",
                (p.player_id, p.name, p.team, p.position, p.created_at),
            )

    def get_players(self, team: Optional[str] = None) -> List[Player]:
        with self._conn() as conn:
            if team:
                rows = conn.execute(
                    "SELECT * FROM players WHERE team=? ORDER BY name", (team,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM players ORDER BY name").fetchall()
            return [Player(**dict(r)) for r in rows]

    def upsert_session(self, s: Session):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sessions
                   (session_id, date, team, session_type, duration_s, notes, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     date=excluded.date, team=excluded.team,
                     session_type=excluded.session_type,
                     duration_s=excluded.duration_s, notes=excluded.notes""",
                (s.session_id, s.date, s.team, s.session_type,
                 s.duration_s, s.notes, s.created_at),
            )

    def get_sessions(self, team: Optional[str] = None,
                     from_date: Optional[str] = None,
                     to_date: Optional[str] = None) -> List[Session]:
        with self._conn() as conn:
            q = "SELECT * FROM sessions WHERE 1=1"
            params: List[Any] = []
            if team:
                q += " AND team=?"; params.append(team)
            if from_date:
                q += " AND date>=?"; params.append(from_date)
            if to_date:
                q += " AND date<=?"; params.append(to_date)
            q += " ORDER BY date DESC"
            rows = conn.execute(q, params).fetchall()
            return [Session(**dict(r)) for r in rows]

    def delete_session(self, session_id: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM player_session_metrics WHERE session_id=?",
                         (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))

    def upsert_metrics(self, m: PlayerSessionMetrics):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO player_session_metrics
                   (session_id, player_id, total_distance_m, relative_distance_m_min,
                    hsr_distance_m, sprint_distance_m, max_speed_m_s, mean_speed_m_s,
                    player_load, accelerations, decelerations,
                    high_intensity_distance_m, duration_s, speed_zones, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, player_id) DO UPDATE SET
                     total_distance_m=excluded.total_distance_m,
                     relative_distance_m_min=excluded.relative_distance_m_min,
                     hsr_distance_m=excluded.hsr_distance_m,
                     sprint_distance_m=excluded.sprint_distance_m,
                     max_speed_m_s=excluded.max_speed_m_s,
                     mean_speed_m_s=excluded.mean_speed_m_s,
                     player_load=excluded.player_load,
                     accelerations=excluded.accelerations,
                     decelerations=excluded.decelerations,
                     high_intensity_distance_m=excluded.high_intensity_distance_m,
                     duration_s=excluded.duration_s,
                     speed_zones=excluded.speed_zones,
                     metadata=excluded.metadata""",
                (m.session_id, m.player_id, m.total_distance_m,
                 m.relative_distance_m_min, m.hsr_distance_m, m.sprint_distance_m,
                 m.max_speed_m_s, m.mean_speed_m_s, m.player_load,
                 m.accelerations, m.decelerations, m.high_intensity_distance_m,
                 m.duration_s, json.dumps(m.speed_zones), json.dumps(m.metadata)),
            )

    def get_metrics(self, session_id: Optional[str] = None,
                    player_id: Optional[str] = None) -> List[PlayerSessionMetrics]:
        with self._conn() as conn:
            q = "SELECT * FROM player_session_metrics WHERE 1=1"
            params: List[Any] = []
            if session_id:
                q += " AND session_id=?"; params.append(session_id)
            if player_id:
                q += " AND player_id=?"; params.append(player_id)
            rows = conn.execute(q, params).fetchall()
            return [PlayerSessionMetrics.from_row(dict(r)) for r in rows]

    def get_player_history(self, player_id: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT s.date, s.session_type, s.team, m.*
                   FROM player_session_metrics m
                   JOIN sessions s ON s.session_id = m.session_id
                   WHERE m.player_id=? ORDER BY s.date""",
                (player_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def summary_stats(self) -> Dict[str, int]:
        with self._conn() as conn:
            n_p = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            n_s = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            n_m = conn.execute(
                "SELECT COUNT(*) FROM player_session_metrics").fetchone()[0]
            return {"players": n_p, "sessions": n_s, "player_sessions": n_m}


# =============================================================================
# SECTION 4: CSV IMPORTER
# =============================================================================

class CSVImporter:
    def __init__(self, db: Database, sample_rate_hz: float = 10.0,
                 zones: Optional[Dict[str, Tuple[float, float]]] = None):
        self.db = db
        self.sample_rate_hz = sample_rate_hz
        self.zones = zones or DEFAULT_ZONES

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        mapping = {}
        cols_lower = {c.lower().strip(): c for c in df.columns}
        for canonical, aliases in COLUMN_ALIASES.items():
            for a in aliases:
                if a in cols_lower:
                    mapping[cols_lower[a]] = canonical
                    break
        return df.rename(columns=mapping)

    def _compute_player_metrics(self, pdf: pd.DataFrame,
                                session_id: str, player_id: str,
                                dt: float) -> PlayerSessionMetrics:
        pdf = pdf.sort_values("timestamp").reset_index(drop=True)

        if "speed" not in pdf.columns:
            if "distance" in pdf.columns:
                pdf["speed"] = pdf["distance"].diff().fillna(0) / dt
            else:
                raise ValueError("CSV must contain 'speed' or 'distance' column")

        speed = pdf["speed"].astype(float).fillna(0).clip(lower=0).values
        n = len(speed)
        duration_s = n * dt
        dist_per_sample = speed * dt
        total_distance = float(dist_per_sample.sum())

        zone_dist = {}
        for zname, (lo, hi) in self.zones.items():
            mask = (speed >= lo) & (speed < hi)
            zone_dist[zname] = float(dist_per_sample[mask].sum())

        hsr = zone_dist.get("zone4_hsr", 0.0)
        sprint = zone_dist.get("zone5_sprint", 0.0)
        max_speed = float(speed.max())
        mean_speed = float(speed.mean())

        if all(c in pdf.columns for c in ("acc_x", "acc_y", "acc_z")):
            ax = pdf["acc_x"].astype(float).fillna(0).values
            ay = pdf["acc_y"].astype(float).fillna(0).values
            az = pdf["acc_z"].astype(float).fillna(0).values
            acc_fwd = np.diff(speed, prepend=speed[0]) / dt
            accels = int(np.sum(acc_fwd > ACC_THRESHOLD))
            decels = int(np.sum(acc_fwd < DEC_THRESHOLD))
            player_load = float(np.sqrt(np.sum(
                np.diff(ax)**2 + np.diff(ay)**2 + np.diff(az)**2
            )))
        else:
            acc_fwd = np.diff(speed, prepend=speed[0]) / dt
            accels = int(np.sum(acc_fwd > ACC_THRESHOLD))
            decels = int(np.sum(acc_fwd < DEC_THRESHOLD))
            player_load = float(np.sqrt(np.sum(np.diff(speed) ** 2)))

        relative_distance = (total_distance / duration_s * 60) if duration_s > 0 else 0.0

        name = str(pdf["player_name"].iloc[0]) if "player_name" in pdf.columns else player_id
        team = str(pdf["team"].iloc[0]) if "team" in pdf.columns else "Senior"
        position = str(pdf["position"].iloc[0]) if "position" in pdf.columns else "Unknown"

        return PlayerSessionMetrics(
            session_id=session_id, player_id=player_id,
            total_distance_m=round(total_distance, 2),
            relative_distance_m_min=round(relative_distance, 2),
            hsr_distance_m=round(hsr, 2),
            sprint_distance_m=round(sprint, 2),
            max_speed_m_s=round(max_speed, 2),
            mean_speed_m_s=round(mean_speed, 2),
            player_load=round(player_load, 2),
            accelerations=accels, decelerations=decels,
            high_intensity_distance_m=round(hsr + sprint, 2),
            duration_s=round(duration_s, 2),
            speed_zones={k: round(v, 2) for k, v in zone_dist.items()},
            metadata={"name": name, "team": team, "position": position, "samples": n},
        )

    def import_file(self, csv_path: str,
                    session_id: Optional[str] = None,
                    session_date: Optional[str] = None,
                    session_type: str = "Training",
                    team: Optional[str] = None) -> Tuple[Session, List[PlayerSessionMetrics]]:
        df = pd.read_csv(csv_path)
        df = self._normalize_columns(df)

        if "timestamp" not in df.columns:
            raise ValueError("CSV must contain a timestamp column")
        if "player_id" not in df.columns:
            raise ValueError("CSV must contain a player_id column")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["player_id"] = df["player_id"].astype(str)

        if session_date is None:
            session_date = df["timestamp"].dt.date.min().isoformat()
        if session_id is None:
            h = hashlib.md5(f"{csv_path}:{session_date}".encode()).hexdigest()[:10]
            session_id = f"ses_{session_date}_{h}"
        if team is None:
            team = df["team"].mode().iloc[0] if "team" in df.columns else "Senior"

        dt = 1.0 / self.sample_rate_hz
        metrics_list: List[PlayerSessionMetrics] = []

        for pid, pdf in df.groupby("player_id"):
            m = self._compute_player_metrics(pdf, session_id, pid, dt)
            metrics_list.append(m)
            self.db.upsert_player(Player(
                player_id=pid,
                name=m.metadata.get("name", pid),
                team=m.metadata.get("team", team),
                position=m.metadata.get("position", "Unknown"),
            ))

        session_duration = float(np.median([m.duration_s for m in metrics_list]))
        session = Session(
            session_id=session_id, date=session_date, team=team,
            session_type=session_type, duration_s=round(session_duration, 2),
        )
        self.db.upsert_session(session)
        for m in metrics_list:
            self.db.upsert_metrics(m)
        return session, metrics_list


# =============================================================================
# SECTION 5: ANALYSIS ENGINES
# =============================================================================

class SessionAnalyzer:
    def __init__(self, db: Database):
        self.db = db

    def _metrics_df(self, team: Optional[str] = None,
                    from_date: Optional[str] = None,
                    to_date: Optional[str] = None) -> pd.DataFrame:
        sessions = self.db.get_sessions(team=team, from_date=from_date, to_date=to_date)
        if not sessions:
            return pd.DataFrame()
        rows = []
        for s in sessions:
            for m in self.db.get_metrics(session_id=s.session_id):
                rows.append({
                    "session_id": s.session_id, "date": s.date,
                    "session_type": s.session_type, "team": s.team,
                    "player_id": m.player_id,
                    "player_name": m.metadata.get("name", m.player_id),
                    "position": m.metadata.get("position", "Unknown"),
                    "total_distance_m": m.total_distance_m,
                    "relative_distance_m_min": m.relative_distance_m_min,
                    "hsr_distance_m": m.hsr_distance_m,
                    "sprint_distance_m": m.sprint_distance_m,
                    "high_intensity_distance_m": m.high_intensity_distance_m,
                    "max_speed_m_s": m.max_speed_m_s,
                    "mean_speed_m_s": m.mean_speed_m_s,
                    "player_load": m.player_load,
                    "accelerations": m.accelerations,
                    "decelerations": m.decelerations,
                    "duration_s": m.duration_s,
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def player_summary(self, player_id: str) -> Dict[str, Any]:
        df = self._metrics_df()
        pdf = df[df["player_id"] == player_id].copy()
        if pdf.empty:
            return {"player_id": player_id, "sessions": 0}
        numeric_cols = [
            "total_distance_m", "relative_distance_m_min", "hsr_distance_m",
            "sprint_distance_m", "high_intensity_distance_m", "max_speed_m_s",
            "mean_speed_m_s", "player_load", "accelerations", "decelerations",
        ]
        stats = {c: {
            "mean": round(float(pdf[c].mean()), 2),
            "median": round(float(pdf[c].median()), 2),
            "std": round(float(pdf[c].std()), 2) if len(pdf) > 1 else 0.0,
            "min": round(float(pdf[c].min()), 2),
            "max": round(float(pdf[c].max()), 2),
        } for c in numeric_cols}
        return {
            "player_id": player_id,
            "player_name": pdf["player_name"].iloc[0],
            "position": pdf["position"].iloc[0],
            "sessions": len(pdf),
            "last_session": pdf["date"].max().date().isoformat(),
            "stats": stats,
        }

    def team_summary(self, team: Optional[str] = None) -> pd.DataFrame:
        df = self._metrics_df(team=team)
        if df.empty:
            return df
        return df.groupby("player_id").agg(
            player_name=("player_name", "first"),
            position=("position", "first"),
            sessions=("session_id", "nunique"),
            avg_distance=("total_distance_m", "mean"),
            avg_relative_distance=("relative_distance_m_min", "mean"),
            avg_hsr=("hsr_distance_m", "mean"),
            avg_sprint=("sprint_distance_m", "mean"),
            avg_hi=("high_intensity_distance_m", "mean"),
            max_speed=("max_speed_m_s", "max"),
            avg_load=("player_load", "mean"),
            avg_accels=("accelerations", "mean"),
            avg_decels=("decelerations", "mean"),
        ).round(2).reset_index()

    def team_benchmarks_by_position(self, team: Optional[str] = None) -> pd.DataFrame:
        df = self._metrics_df(team=team)
        if df.empty:
            return df
        return df.groupby("position").agg(
            n_players=("player_id", "nunique"),
            avg_distance=("total_distance_m", "mean"),
            avg_hsr=("hsr_distance_m", "mean"),
            avg_sprint=("sprint_distance_m", "mean"),
            avg_load=("player_load", "mean"),
            avg_max_speed=("max_speed_m_s", "mean"),
        ).round(2).reset_index()

    def session_snapshot(self, session_id: str) -> pd.DataFrame:
        df = self._metrics_df()
        sdf = df[df["session_id"] == session_id].copy()
        if not sdf.empty:
            sdf = sdf.sort_values("total_distance_m", ascending=False)
        return sdf


class ACWREngine:
    def __init__(self, db: Database, load_col: str = "player_load"):
        self.db = db
        self.load_col = load_col

    def compute_for_player(self, player_id: str,
                           as_of_date: Optional[str] = None) -> List[ACWRResult]:
        history = self.db.get_player_history(player_id)
        if not history:
            return []
        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date").set_index("date")
        daily = df[[self.load_col]].groupby(df.index).sum()
        full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_idx, fill_value=0)

        acute = daily[self.load_col].rolling(7, min_periods=1).sum()
        chronic_21 = daily[self.load_col].rolling(21, min_periods=7).mean()
        chronic_28 = daily[self.load_col].rolling(28, min_periods=7).mean()
        chronic_35 = daily[self.load_col].rolling(35, min_periods=7).mean()

        results: List[ACWRResult] = []
        for d in daily.index:
            if as_of_date and d.date().isoformat() != as_of_date:
                continue
            a = float(acute.loc[d])
            c21 = float(chronic_21.loc[d]) if not pd.isna(chronic_21.loc[d]) else 0
            c28 = float(chronic_28.loc[d]) if not pd.isna(chronic_28.loc[d]) else 0
            c35 = float(chronic_35.loc[d]) if not pd.isna(chronic_35.loc[d]) else 0

            def ratio(a, c):
                return round(a / c, 2) if c > 0 else 0.0

            r21, r28, r35 = ratio(a, c21), ratio(a, c28), ratio(a, c35)
            primary = r28 if r28 > 0 else r21
            if primary == 0:
                band = "no_data"
            elif primary < 0.80:
                band = "underloading"
            elif primary <= 1.30:
                band = "sweet_spot"
            elif primary <= 1.50:
                band = "caution"
            else:
                band = "high_risk"

            results.append(ACWRResult(
                player_id=player_id, date=d.date().isoformat(),
                acute_load=round(a, 2),
                chronic_load_21=round(c21, 2),
                chronic_load_28=round(c28, 2),
                chronic_load_35=round(c35, 2),
                ratio_7_21=r21, ratio_7_28=r28, ratio_7_35=r35,
                risk_band=band,
            ))
        return results

    def squad_acwr_snapshot(self, team: Optional[str] = None,
                            as_of_date: Optional[str] = None) -> pd.DataFrame:
        players = self.db.get_players(team=team)
        rows = []
        for p in players:
            res = self.compute_for_player(p.player_id, as_of_date=as_of_date)
            if res:
                latest = res[-1]
                rows.append({
                    "player_id": p.player_id, "player_name": p.name,
                    "position": p.position, "date": latest.date,
                    "acute_7d": latest.acute_load,
                    "chronic_28d": latest.chronic_load_28,
                    "acwr_7_28": latest.ratio_7_28,
                    "risk_band": latest.risk_band,
                })
        return pd.DataFrame(rows)


class SpeedZoneEngine:
    def __init__(self, db: Database):
        self.db = db

    def team_zone_distribution(self, team: Optional[str] = None) -> pd.DataFrame:
        sessions = self.db.get_sessions(team=team)
        rows = []
        for s in sessions:
            for m in self.db.get_metrics(session_id=s.session_id):
                row = {
                    "player_id": m.player_id,
                    "player_name": m.metadata.get("name", m.player_id),
                    "session_id": s.session_id, "date": s.date,
                }
                for z, d in m.speed_zones.items():
                    row[z] = d
                row["total"] = m.total_distance_m
                rows.append(row)
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        zone_cols = [c for c in df.columns if c.startswith("zone")]
        for c in zone_cols:
            df[c + "_pct"] = (df[c] / df["total"].replace(0, np.nan) * 100).round(1)
        return df


# =============================================================================
# SECTION 6: FORECASTING
# =============================================================================

class SessionForecaster:
    def __init__(self, db: Database, alpha: float = 0.35, beta: float = 0.1,
                 load_col: str = "player_load"):
        self.db = db
        self.alpha = alpha
        self.beta = beta
        self.load_col = load_col

    def _holt_forecast(self, series: List[float], horizon: int = 1):
        if len(series) < 2:
            v = series[0] if series else 0
            return v, v * 0.8, v * 1.2
        level = series[0]
        trend = series[1] - series[0]
        for y in series[1:]:
            new_level = self.alpha * y + (1 - self.alpha) * (level + trend)
            new_trend = self.beta * (new_level - level) + (1 - self.beta) * trend
            level, trend = new_level, new_trend
        forecast = level + horizon * trend
        preds = []
        lv, tr = series[0], series[1] - series[0]
        for y in series[1:]:
            preds.append(lv + tr)
            lv = self.alpha * y + (1 - self.alpha) * (lv + tr)
            tr = self.beta * (lv - (lv - tr)) + (1 - self.beta) * tr
        residuals = np.array(series[1:]) - np.array(preds)
        std = float(np.std(residuals)) if len(residuals) > 1 else forecast * 0.15
        return max(0, forecast), max(0, forecast - 1.96 * std), forecast + 1.96 * std

    def _session_type_factor(self, session_type: str) -> float:
        t = session_type.lower()
        if "match" in t or t == "md":
            return 1.35
        if "md-1" in t or "md+1" in t:
            return 0.75
        if "recovery" in t:
            return 0.45
        if "tactical" in t:
            return 0.85
        return 1.0

    def forecast_player(self, player_id: str, target_date: str,
                        session_type: str = "Training") -> Forecast:
        history = self.db.get_player_history(player_id)
        if not history:
            return Forecast(player_id=player_id, target_date=target_date,
                            predicted_player_load=0, predicted_distance_m=0,
                            predicted_hsr_m=0, confidence_low=0,
                            confidence_high=0, method="no_data")
        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        load_series = df[self.load_col].tolist()
        dist_series = df["total_distance_m"].tolist()
        hsr_series = df["hsr_distance_m"].tolist()

        load_fc, load_lo, load_hi = self._holt_forecast(load_series)
        dist_fc, _, _ = self._holt_forecast(dist_series)
        hsr_fc, _, _ = self._holt_forecast(hsr_series)

        factor = self._session_type_factor(session_type)
        return Forecast(
            player_id=player_id, target_date=target_date,
            predicted_player_load=round(load_fc * factor, 2),
            predicted_distance_m=round(dist_fc * factor, 2),
            predicted_hsr_m=round(hsr_fc * factor, 2),
            confidence_low=round(load_lo * factor, 2),
            confidence_high=round(load_hi * factor, 2),
            method="holt_linear_es",
        )

    def forecast_squad(self, target_date: str, session_type: str = "Training",
                       team: Optional[str] = None) -> pd.DataFrame:
        players = self.db.get_players(team=team)
        rows = []
        for p in players:
            f = self.forecast_player(p.player_id, target_date, session_type)
            rows.append({
                "player_id": p.player_id, "player_name": p.name,
                "position": p.position, "target_date": target_date,
                "session_type": session_type,
                "predicted_load": f.predicted_player_load,
                "predicted_distance_m": f.predicted_distance_m,
                "predicted_hsr_m": f.predicted_hsr_m,
                "ci_low": f.confidence_low, "ci_high": f.confidence_high,
            })
        return pd.DataFrame(rows)


# =============================================================================
# SECTION 7: REPORTS
# =============================================================================

class ReportGenerator:
    def __init__(self, db: Database):
        self.db = db
        self.analyzer = SessionAnalyzer(db)
        self.acwr = ACWREngine(db)

    def session_report(self, session_id: str) -> str:
        sessions = self.db.get_sessions()
        s = next((x for x in sessions if x.session_id == session_id), None)
        if not s:
            return "Session not found."
        df = self.analyzer.session_snapshot(session_id)
        if df.empty:
            return "No metrics for this session."

        lines = [
            f"# Session Report: {s.session_id}",
            f"- **Date**: {s.date}",
            f"- **Team**: {s.team}",
            f"- **Type**: {s.session_type}",
            f"- **Duration**: {s.duration_s/60:.1f} min",
            f"- **Players**: {len(df)}", "",
            "## Squad Summary", "",
            "| Player | Pos | Dist (m) | Rel (m/min) | HSR (m) | Sprint (m) | Max Spd (m/s) | Load |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in df.iterrows():
            lines.append(
                f"| {r['player_name']} | {r['position']} "
                f"| {r['total_distance_m']:.0f} | {r['relative_distance_m_min']:.1f} "
                f"| {r['hsr_distance_m']:.0f} | {r['sprint_distance_m']:.0f} "
                f"| {r['max_speed_m_s']:.1f} | {r['player_load']:.0f} |"
            )
        lines.extend([
            "", "## Squad Averages",
            f"- Mean distance: **{df['total_distance_m'].mean():.0f} m**",
            f"- Mean HSR: **{df['hsr_distance_m'].mean():.0f} m**",
            f"- Mean sprint: **{df['sprint_distance_m'].mean():.0f} m**",
            f"- Mean PlayerLoad: **{df['player_load'].mean():.0f} AU**",
            f"- Mean max speed: **{df['max_speed_m_s'].mean():.1f} m/s**",
        ])
        for col, label in [("total_distance_m", "distance"),
                           ("player_load", "load")]:
            mean, std = df[col].mean(), df[col].std()
            outliers = df[df[col] > mean + 1.5 * std]
            if len(outliers) > 0:
                lines.extend(["", f"## {label.capitalize()} Outliers (>+1.5 SD)"])
                for _, r in outliers.iterrows():
                    lines.append(f"- {r['player_name']}: {r[col]:.0f}")
        return "\n".join(lines)

    def player_report(self, player_id: str) -> str:
        summary = self.analyzer.player_summary(player_id)
        if summary.get("sessions", 0) == 0:
            return "No data for this player."
        acwr_results = self.acwr.compute_for_player(player_id)
        latest_acwr = acwr_results[-1] if acwr_results else None

        lines = [
            f"# Player Report: {summary['player_name']}",
            f"- **Position**: {summary['position']}",
            f"- **Sessions tracked**: {summary['sessions']}",
            f"- **Last session**: {summary['last_session']}", "",
            "## Career Averages", "",
            "| Metric | Mean | Median | SD | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for metric, s in summary["stats"].items():
            lines.append(
                f"| {metric} | {s['mean']} | {s['median']} "
                f"| {s['std']} | {s['min']} | {s['max']} |"
            )
        if latest_acwr:
            lines.extend([
                "", "## ACWR Status",
                f"- **Date**: {latest_acwr.date}",
                f"- **Acute (7d)**: {latest_acwr.acute_load:.1f}",
                f"- **Chronic (28d)**: {latest_acwr.chronic_load_28:.1f}",
                f"- **ACWR (7:28)**: {latest_acwr.ratio_7_28:.2f}",
                f"- **Risk band**: {latest_acwr.risk_band}",
            ])
        return "\n".join(lines)


# =============================================================================
# SECTION 8: SAMPLE DATA GENERATOR (built-in for instant demo)
# =============================================================================

SAMPLE_PLAYERS = [
    ("P01", "Alex Morgan",    "Senior", "Forward"),
    ("P02", "Sam Kerr",       "Senior", "Forward"),
    ("P03", "Lucy Bronze",    "Senior", "Defender"),
    ("P04", "Leah Williamson","Senior", "Defender"),
    ("P05", "Keira Walsh",    "Senior", "Midfielder"),
    ("P06", "Aitana Bonmati", "Senior", "Midfielder"),
    ("P07", "Mary Earps",     "Senior", "Goalkeeper"),
    ("P08", "Jess Carter",    "Senior", "Defender"),
    ("P09", "Ella Toone",     "Senior", "Midfielder"),
    ("P10", "Lauren Hemp",    "Senior", "Forward"),
    ("P11", "Rachel Daly",    "Senior", "Defender"),
]

PROFILES = {
    "Forward":    {"dist": 10500, "max_spd": 8.8},
    "Midfielder": {"dist": 11500, "max_spd": 8.2},
    "Defender":   {"dist": 9800,  "max_spd": 8.5},
    "Goalkeeper": {"dist": 3500,  "max_spd": 6.5},
}


def generate_sample_csv(date: datetime, session_type: str = "Training") -> pd.DataFrame:
    np.random.seed(int(date.timestamp()) % 10000)
    rows = []
    sample_rate = 10
    duration_min = {"Match": 90, "Recovery": 30}.get(session_type, 75)

    for pid, name, team, pos in SAMPLE_PLAYERS:
        prof = PROFILES[pos]
        total_dist = prof["dist"] * np.random.uniform(0.85, 1.15)
        n_samples = duration_min * 60 * sample_rate
        times = pd.date_range(date, periods=n_samples, freq=f"{1/sample_rate}S")

        base_speed = total_dist / (duration_min * 60)
        speeds = np.random.exponential(base_speed * 0.7, n_samples)
        speeds = np.clip(speeds, 0, 3.0)

        n_hsr = int(duration_min * np.random.uniform(15, 30))
        for _ in range(n_hsr):
            start = np.random.randint(0, n_samples - 100)
            length = np.random.randint(30, 120)
            speeds[start:start+length] = np.random.uniform(5.5, 6.9, length)

        n_sprints = int(duration_min * np.random.uniform(3, 8))
        for _ in range(n_sprints):
            start = np.random.randint(0, n_samples - 50)
            length = np.random.randint(15, 50)
            speeds[start:start+length] = np.random.uniform(7.0, prof["max_spd"], length)

        speeds = np.clip(speeds, 0, 10)
        ax = np.random.normal(0, 1.5, n_samples)
        ay = np.random.normal(0, 1.5, n_samples)
        az = np.random.normal(9.81, 1.5, n_samples)

        rows.append(pd.DataFrame({
            "timestamp": times, "player_id": pid, "player_name": name,
            "team": team, "position": pos,
            "speed": np.round(speeds, 3),
            "acc_x": np.round(ax, 3), "acc_y": np.round(ay, 3),
            "acc_z": np.round(az, 3),
            "heart_rate": np.random.randint(110, 185, n_samples),
        }))
    return pd.concat(rows, ignore_index=True)


def load_demo_data(db: Database, n_weeks: int = 6, progress_callback=None):
    """Generate and import n_weeks of synthetic sessions."""
    importer = CSVImporter(db, sample_rate_hz=10.0)
    start = datetime(2026, 8, 3)
    total = n_weeks * 6  # 6 training days/week
    done = 0
    for i in range(n_weeks * 7):
        d = start + timedelta(days=i)
        if d.weekday() == 6:
            continue
        if i % 7 == 5:
            stype = "Match"
        elif i % 7 == 6:
            stype = "Recovery"
        else:
            stype = "Training"
        df = generate_sample_csv(d, stype)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w") as tmp:
            df.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
        try:
            importer.import_file(tmp_path, session_type=stype)
        finally:
            os.unlink(tmp_path)
        done += 1
        if progress_callback:
            progress_callback(done / total)


# =============================================================================
# SECTION 9: STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Sport GPS Analysis", layout="wide", page_icon="🏃")


@st.cache_resource
def get_db():
    return Database(DB_PATH)


def render_import(db: Database):
    st.subheader("📥 Import GPS CSV")
    st.caption("Supports Catapult, STATSports, GPSports, and custom formats.")

    # Demo data loader
    with st.expander("⚡ No CSV? Load 6 weeks of demo data instantly", expanded=False):
        st.write("This generates realistic synthetic GPS data for 11 players across 6 weeks.")
        if st.button("🚀 Load demo data"):
            progress_bar = st.progress(0)
            status = st.empty()
            def cb(p):
                progress_bar.progress(p)
                status.write(f"Importing... {int(p*100)}%")
            load_demo_data(db, n_weeks=6, progress_callback=cb)
            status.success("✅ Demo data loaded! Refresh to see results.")
            st.rerun()

    st.markdown("---")
    uploaded = st.file_uploader("Upload GPS CSV", type=["csv"])
    col1, col2, col3 = st.columns(3)
    with col1:
        session_date = st.date_input("Session date").isoformat()
    with col2:
        session_type = st.selectbox("Session type",
                                    ["Training", "Match", "MD-1", "MD+1",
                                     "Recovery", "Tactical"])
    with col3:
        team = st.text_input("Team", value="Senior")
    sample_rate = st.number_input("Sample rate (Hz)", value=10.0, step=1.0)

    if uploaded and st.button("Import", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            importer = CSVImporter(db, sample_rate_hz=sample_rate)
            session, metrics = importer.import_file(
                tmp_path, session_date=session_date,
                session_type=session_type, team=team,
            )
            st.success(f"✅ Imported `{session.session_id}` — {len(metrics)} players")
            df = pd.DataFrame([m.__dict__ for m in metrics])
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Import failed: {e}")
        finally:
            os.unlink(tmp_path)


def render_session(db: Database):
    st.subheader("📊 Session Analysis")
    sessions = db.get_sessions()
    if not sessions:
        st.info("No sessions yet. Import a CSV or load demo data first.")
        return
    sel = st.selectbox("Select session",
                       [f"{s.date} — {s.session_type} ({s.team})" for s in sessions])
    s = sessions[[f"{x.date} — {x.session_type} ({x.team})" for x in sessions].index(sel)]
    analyzer = SessionAnalyzer(db)
    df = analyzer.session_snapshot(s.session_id)
    if df.empty:
        st.warning("No metrics for this session.")
        return

    st.dataframe(df, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(df, x="player_name", y="total_distance_m",
                     title="Total Distance (m)", color="position")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(df, x="player_name",
                     y=["hsr_distance_m", "sprint_distance_m"],
                     title="High-Intensity Distance (m)", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df, x="total_distance_m", y="player_load",
                     hover_name="player_name", color="position",
                     title="Distance vs PlayerLoad")
    st.plotly_chart(fig, use_container_width=True)

    # Speed zone breakdown
    zone_df = []
    for _, r in df.iterrows():
        metrics = db.get_metrics(session_id=s.session_id, player_id=r["player_id"])
        if metrics:
            m = metrics[0]
            row = {"player": r["player_name"]}
            row.update(m.speed_zones)
            zone_df.append(row)
    if zone_df:
        zdf = pd.DataFrame(zone_df).set_index("player")
        fig = px.bar(zdf, barmode="stack",
                     title="Distance by Speed Zone (m)",
                     color_discrete_map={
                         "zone1_walk": "#A8D5BA", "zone2_jog": "#7FC8A9",
                         "zone3_run": "#F4D35E", "zone4_hsr": "#EE964B",
                         "zone5_sprint": "#F95738",
                     })
        st.plotly_chart(fig, use_container_width=True)


def render_players(db: Database):
    st.subheader("👤 Player & Team Analysis")
    analyzer = SessionAnalyzer(db)
    team_df = analyzer.team_summary()
    if team_df.empty:
        st.info("No data yet.")
        return

    st.markdown("### Squad Overview")
    st.dataframe(team_df, use_container_width=True)

    bench = analyzer.team_benchmarks_by_position()
    if not bench.empty:
        st.markdown("### Positional Benchmarks")
        st.dataframe(bench, use_container_width=True)

    st.markdown("---")
    st.markdown("### Individual Player Drill-Down")
    pid = st.selectbox("Select player",
                       team_df["player_id"].tolist(),
                       format_func=lambda x: team_df[team_df["player_id"]==x]["player_name"].iloc[0])
    summary = analyzer.player_summary(pid)
    if summary.get("sessions", 0) > 0:
        history = pd.DataFrame(db.get_player_history(pid))
        history["date"] = pd.to_datetime(history["date"])

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(history, x="date", y="total_distance_m",
                          title=f"{summary['player_name']} — Total Distance")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(history, x="date", y="player_load",
                          title=f"{summary['player_name']} — PlayerLoad")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.line(history, x="date", y="hsr_distance_m",
                          title=f"{summary['player_name']} — HSR Distance")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.line(history, x="date", y="max_speed_m_s",
                          title=f"{summary['player_name']} — Max Speed")
            st.plotly_chart(fig, use_container_width=True)


def render_acwr(db: Database):
    st.subheader("⚖️ Acute / Chronic Workload Ratio")
    st.caption("Risk bands: 🟢 Sweet spot (0.80–1.30) · 🟠 Underloading (<0.80) · 🟡 Caution (1.30–1.50) · 🔴 High risk (>1.50)")

    acwr_eng = ACWREngine(db)
    snap = acwr_eng.squad_acwr_snapshot()
    if snap.empty:
        st.info("Need at least 7 days of data per player to compute ACWR.")
        return

    st.dataframe(snap, use_container_width=True)

    color_map = {
        "sweet_spot": "#2ecc71", "underloading": "#e67e22",
        "caution": "#f1c40f", "high_risk": "#e74c3c", "no_data": "#95a5a6",
    }
    fig = px.scatter(snap, x="acwr_7_28", y="player_name",
                     color="risk_band", color_discrete_map=color_map,
                     title="Squad ACWR (7:28) — latest snapshot",
                     hover_data=["acute_7d", "chronic_28d", "position"])
    fig.add_vline(x=0.80, line_dash="dash", line_color="#e67e22")
    fig.add_vline(x=1.30, line_dash="dash", line_color="#f1c40f")
    fig.add_vline(x=1.50, line_dash="dash", line_color="#e74c3c")
    st.plotly_chart(fig, use_container_width=True)

    # ACWR over time for selected player
    players = db.get_players()
    if players:
        pname = st.selectbox("View ACWR timeline for", [p.name for p in players])
        pid = next(p.player_id for p in players if p.name == pname)
        results = acwr_eng.compute_for_player(pid)
        if results:
            rdf = pd.DataFrame([asdict(r) for r in results])
            rdf["date"] = pd.to_datetime(rdf["date"])
            fig = px.line(rdf, x="date", y=["ratio_7_21", "ratio_7_28", "ratio_7_35"],
                          title=f"ACWR Timeline — {pname}")
            fig.add_hline(y=0.80, line_dash="dash", line_color="#e67e22")
            fig.add_hline(y=1.30, line_dash="dash", line_color="#f1c40f")
            fig.add_hline(y=1.50, line_dash="dash", line_color="#e74c3c")
            st.plotly_chart(fig, use_container_width=True)


def render_forecast(db: Database):
    st.subheader("🔮 Session Forecasting")
    st.caption("Holt linear exponential smoothing + session-type adjustment + 95% confidence interval.")

    col1, col2 = st.columns(2)
    with col1:
        target_date = st.date_input("Target date",
                                    value=datetime.today() + timedelta(days=1)).isoformat()
    with col2:
        session_type = st.selectbox("Planned session type",
                                    ["Training", "Match", "MD-1", "Tactical", "Recovery"])

    if st.button("Generate forecast", type="primary"):
        fc = SessionForecaster(db)
        df = fc.forecast_squad(target_date, session_type)
        if df.empty:
            st.info("No player history to forecast from.")
            return

        st.dataframe(df, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["player_name"], y=df["predicted_load"],
            error_y=dict(type="data", symmetric=False,
                         array=df["ci_high"] - df["predicted_load"],
                         arrayminus=df["predicted_load"] - df["ci_low"]),
            name="Predicted Load", marker_color="#3498db",
        ))
        fig.update_layout(title=f"Predicted PlayerLoad — {target_date} ({session_type})",
                          xaxis_title="Player", yaxis_title="PlayerLoad (AU)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(df, x="player_name",
                      y=["predicted_distance_m", "predicted_hsr_m"],
                      barmode="group",
                      title="Predicted Distance & HSR")
        st.plotly_chart(fig2, use_container_width=True)


def render_reports(db: Database):
    st.subheader("📄 Reports")
    rep = ReportGenerator(db)
    rtype = st.radio("Report type", ["Session", "Player"], horizontal=True)

    if rtype == "Session":
        sessions = db.get_sessions()
        if not sessions:
            st.info("No sessions yet.")
            return
        sel = st.selectbox("Session",
                           [f"{s.date} — {s.session_type}" for s in sessions])
        idx = [f"{x.date} — {x.session_type}" for x in sessions].index(sel)
        md = rep.session_report(sessions[idx].session_id)
        st.markdown(md)
        st.download_button("⬇️ Download .md", md,
                           file_name=f"session_report_{sessions[idx].session_id}.md")
    else:
        players = db.get_players()
        if not players:
            st.info("No players yet.")
            return
        sel = st.selectbox("Player", [p.name for p in players])
        idx = [p.name for p in players].index(sel)
        md = rep.player_report(players[idx].player_id)
        st.markdown(md)
        st.download_button("⬇️ Download .md", md,
                           file_name=f"player_report_{players[idx].player_id}.md")


def main():
    db = get_db()
    st.title("🏃 Sport GPS Analysis Tool")

    stats = db.summary_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", stats["players"])
    c2.metric("Sessions", stats["sessions"])
    c3.metric("Player-Sessions", stats["player_sessions"])

    tab_import, tab_session, tab_players, tab_acwr, tab_forecast, tab_reports = st.tabs(
        ["📥 Import", "📊 Session", "👤 Players", "⚖️ ACWR", "🔮 Forecast", "📄 Reports"]
    )

    with tab_import:    render_import(db)
    with tab_session:   render_session(db)
    with tab_players:   render_players(db)
    with tab_acwr:      render_acwr(db)
    with tab_forecast:  render_forecast(db)
    with tab_reports:   render_reports(db)

    # Sidebar info
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
        **Sport GPS Analysis Tool**
        - CSV import with auto-schema detection
        - SQLite persistence
        - ACWR (7/21/28/35-day)
        - Holt forecasting
        - Session & player reports
        """)
        st.markdown("### Speed Zones (m/s)")
        st.markdown("""
        | Zone | Range |
        |---|---|
        | Walk | 0.0 – 1.9 |
        | Jog | 2.0 – 3.4 |
        | Run | 3.5 – 5.4 |
        | HSR | 5.5 – 6.9 |
        | Sprint | ≥ 7.0 |
        """)
        if st.button("🗑️ Reset database"):
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
            st.cache_resource.clear()
            st.rerun()


if __name__ == "__main__":
    main()
else:
    # When imported as a module by Streamlit Cloud
    main()