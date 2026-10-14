"""Remote work and the gap between reported and diary-measured hours (ATUS).

Design (see RESEARCH_MEMO.md, Project 1):
  gap_i = usual weekly hours (recall) - 7 x diary work hours (time diary)
  Exposure_o = occupation-level telework exposure, fixed pre-2020
  Event study: gap_iot = sum_k b_k 1[year=k] x Exposure_o + a_o + d_t + X_i'g + e
  SEs clustered by occupation; base year 2019.

Outputs CSV tables to --out-dir. Prints nothing that looks like a result
unless real ATUS data are supplied.

Usage:
  python analysis_hours_gap.py --data data/atus_personday.parquet --out-dir output
      [--exposure dingel_neiman_by_census_occ.csv]   # cols: occ, exposure
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import pyfixest as pf

PRE = range(2015, 2020)
BASE = 2019
COVID_GAP = {2020}  # 2020 ATUS has no data for mid-March to mid-May; dropped


def exposure_from_atus(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-2020 share of occupation work minutes performed at home."""
    pre = df[df["TUYEAR"].isin(PRE) & (df["work_min"] > 0)]
    g = pre.assign(wh=pre["work_home_min"] * pre["wt"], w=pre["work_min"] * pre["wt"])
    e = g.groupby("TEIO1OCD")[["wh", "w"]].sum()
    n = pre.groupby("TEIO1OCD").size()
    e = (e["wh"] / e["w"]).where(n >= 30)  # thin cells are too noisy to rank
    return e.rename("exposure").rename_axis("occ").reset_index().dropna()


def prepare(df: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    s = df[df["employed"] & df["TEIO1OCD"].gt(0) & ~df["TUYEAR"].isin(COVID_GAP)
           & (df["TRHOLIDAY"] == 0)].copy()
    s = s.merge(exposure, left_on="TEIO1OCD", right_on="occ", how="inner")
    s["exposure_z"] = (s["exposure"] - s["exposure"].mean()) / s["exposure"].std()
    s["post"] = (s["TUYEAR"] >= 2021).astype(int)
    return s


def run(s: pd.DataFrame, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. Descriptive: weighted gap by year and exposure tercile
    s["tercile"] = pd.qcut(s["exposure"].rank(method="first"), 3, labels=["low", "mid", "high"])
    desc = (s.dropna(subset=["gap"])
              .groupby(["TUYEAR", "tercile"], observed=True)
              .apply(lambda d: pd.Series({
                  "usual": np.average(d["usual_hrs"], weights=d["wt"]),
                  "diary": np.average(d["diary_wk_hrs"], weights=d["wt"]),
                  "gap": np.average(d["gap"], weights=d["wt"]),
                  "n": len(d)}), include_groups=False)
              .reset_index())
    desc.to_csv(out / "gap_by_year_tercile.csv", index=False)

    ctrl = "C(TUDIARYDAY) + C(TUMONTH)"
    fe = "TEIO1OCD + TUYEAR"
    vc = {"CRV1": "TEIO1OCD"}

    # 2. Event study on the gap and on each component
    for y in ["gap", "usual_hrs", "diary_wk_hrs"]:
        m = pf.feols(f"{y} ~ i(TUYEAR, exposure_z, ref={BASE}) + {ctrl} | {fe}",
                     data=s, weights="wt", vcov=vc)
        t = m.tidy().reset_index()
        t["outcome"] = y
        rows.append(t[t["Coefficient"].str.contains("TUYEAR")])
    pd.concat(rows).to_csv(out / "event_study.csv", index=False)

    # 3. Pooled DiD, with and without state-by-year FE (local labour-market shocks)
    specs = {
        "base": f"gap ~ post:exposure_z + {ctrl} | {fe}",
        "state_year": f"gap ~ post:exposure_z + {ctrl} | TEIO1OCD + GESTFIPS^TUYEAR",
        "hours_vary_outcome": f"hours_vary ~ post:exposure_z + {ctrl} | {fe}",
        # Mechanism: fragmentation of the workday
        "episodes": f"work_episodes ~ post:exposure_z + {ctrl} | {fe}",
    }
    did = []
    for name, f in specs.items():
        d = s if name in ("hours_vary_outcome",) else s.dropna(subset=["gap"])
        m = pf.feols(f, data=d, weights="wt", vcov=vc)
        t = m.tidy().reset_index()
        t = t[t["Coefficient"].str.contains("exposure_z")]
        t["spec"], t["n"] = name, m._N
        did.append(t)
    pd.concat(did).to_csv(out / "did.csv", index=False)

    # 4. Placebo: occupations with ~zero pre-period home work should show no change
    lo = s[s["exposure"] <= s["exposure"].quantile(0.25)].dropna(subset=["gap"])
    m = pf.feols(f"gap ~ post + {ctrl} | TEIO1OCD", data=lo, weights="wt", vcov=vc)
    m.tidy().reset_index().to_csv(out / "placebo_low_exposure.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("output"))
    p.add_argument("--exposure", type=pathlib.Path, help="external occ-level exposure CSV")
    a = p.parse_args()
    df = pd.read_parquet(a.data)
    exp = pd.read_csv(a.exposure) if a.exposure else exposure_from_atus(df)
    run(prepare(df, exp), a.out_dir)
    print(f"tables written to {a.out_dir}")
