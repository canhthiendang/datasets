"""Build a person-day analysis file from the BLS ATUS multi-year public-use files.

Downloads (or reads from --raw-dir) the 2003-2024 respondent, activity and
ATUS-CPS files and writes one row per diary (respondent-day) with:
  - work minutes (activity codes 0501xx), split by location (home vs elsewhere)
  - number of work episodes (fragmentation)
  - usual weekly hours reported in the ATUS interview (TEHRUSLT)
  - occupation, state, diary-day type, holiday flag, survey weight

Variable names follow the BLS ATUS codebook. The script fails loudly if an
expected column is missing, so codebook changes surface immediately.

Usage:
  python build_atus.py --raw-dir data/raw --out data/atus_personday.parquet [--download]
"""
import argparse
import io
import pathlib
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BLS = "https://www.bls.gov/tus/datafiles/"
VINTAGE = "0324"  # 2003-2024 multi-year files; update when BLS adds a year
FILES = {"resp": "atusresp", "act": "atusact", "cps": "atuscps"}

RESP_COLS = ["TUCASEID", "TUYEAR", "TUMONTH", "TUDIARYDAY", "TRHOLIDAY",
             "TELFS", "TEHRUSLT", "TEIO1OCD", "TUFNWGTP", "TU20FWGT"]
ACT_COLS = ["TUCASEID", "TRCODEP", "TUACTDUR24", "TEWHERE"]
CPS_COLS = ["TUCASEID", "TULINENO", "GESTFIPS"]

HOME = 1  # TEWHERE: respondent's home or yard


def fetch(stem: str, raw_dir: pathlib.Path, download: bool) -> pd.DataFrame:
    dat = raw_dir / f"{stem}_{VINTAGE}.dat"
    if not dat.exists():
        if not download:
            raise FileNotFoundError(f"{dat} missing; rerun with --download")
        req = urllib.request.Request(
            f"{BLS}{stem}-{VINTAGE}.zip",
            headers={"User-Agent": "academic research (atus build script)"})
        with urllib.request.urlopen(req, timeout=300) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract(dat.name, raw_dir)
    return pd.read_csv(dat, low_memory=False)


def require(df: pd.DataFrame, cols: list, name: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    # TU20FWGT exists only in vintages that include 2020+
    missing = [c for c in missing if c != "TU20FWGT"]
    if missing:
        raise KeyError(f"{name}: missing columns {missing}; check codebook")
    return df[[c for c in cols if c in df.columns]].copy()


def build(raw_dir: pathlib.Path, download: bool) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    resp = require(fetch(FILES["resp"], raw_dir, download), RESP_COLS, "resp")
    act = require(fetch(FILES["act"], raw_dir, download), ACT_COLS, "act")
    cps = require(fetch(FILES["cps"], raw_dir, download), CPS_COLS, "cps")

    act["work"] = act["TRCODEP"].between(50100, 50199)
    act["work_home"] = act["work"] & (act["TEWHERE"] == HOME)
    act["dur_work"] = np.where(act["work"], act["TUACTDUR24"], 0)
    act["dur_work_home"] = np.where(act["work_home"], act["TUACTDUR24"], 0)
    day = act.groupby("TUCASEID").agg(
        work_min=("dur_work", "sum"),
        work_home_min=("dur_work_home", "sum"),
        work_episodes=("work", "sum"),
        total_min=("TUACTDUR24", "sum"),
    ).reset_index()

    # Respondent line number is 1 in the ATUS-CPS roster
    state = cps.loc[cps["TULINENO"] == 1, ["TUCASEID", "GESTFIPS"]]

    df = resp.merge(day, on="TUCASEID", how="left").merge(state, on="TUCASEID", how="left")
    df[["work_min", "work_home_min", "work_episodes"]] = (
        df[["work_min", "work_home_min", "work_episodes"]].fillna(0))

    # 2020+ weights are TU20FWGT; earlier years TUFNWGTP (see BLS 2020 user notes)
    w20 = df["TU20FWGT"] if "TU20FWGT" in df else pd.Series(np.nan, index=df.index)
    df["wt"] = np.where((df["TUYEAR"] >= 2020) & (w20 > 0), w20, df["TUFNWGTP"])
    # Normalise so each year carries equal total weight in pooled regressions
    df["wt"] = df["wt"] / df.groupby("TUYEAR")["wt"].transform("sum")

    df["employed"] = df["TELFS"].isin([1, 2])
    df["weekend"] = df["TUDIARYDAY"].isin([1, 7])
    df["usual_hrs"] = df["TEHRUSLT"].where(df["TEHRUSLT"] >= 0)  # -4 = hours vary
    df["hours_vary"] = df["TEHRUSLT"] == -4
    # Diary-implied weekly hours: ATUS weights balance weekdays/weekends, so
    # E[7 * daily hours] equals mean weekly hours in the weighted population.
    df["diary_wk_hrs"] = 7 * df["work_min"] / 60
    df["gap"] = df["usual_hrs"] - df["diary_wk_hrs"]
    df["any_home_work"] = df["work_home_min"] > 0
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=pathlib.Path, default=pathlib.Path("data/raw"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/atus_personday.parquet"))
    p.add_argument("--download", action="store_true")
    a = p.parse_args()
    out = build(a.raw_dir, a.download)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"wrote {len(out):,} diaries to {a.out}")
