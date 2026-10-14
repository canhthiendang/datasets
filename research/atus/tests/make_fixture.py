"""Write SYNTHETIC files with the ATUS layout so the pipeline can be smoke-tested
without network access. Numbers produced from these files are meaningless."""
import pathlib
import sys

import numpy as np
import pandas as pd

raw = pathlib.Path(sys.argv[1])
raw.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
n = 20000
years = rng.choice([y for y in range(2012, 2025)], n)
occ = rng.choice(np.arange(1000, 1060) * 10, n)
resp = pd.DataFrame({
    "TUCASEID": np.arange(n) + 20120101000000,
    "TUYEAR": years, "TUMONTH": rng.integers(1, 13, n),
    "TUDIARYDAY": rng.integers(1, 8, n), "TRHOLIDAY": (rng.random(n) < .02).astype(int),
    "TELFS": rng.choice([1, 1, 1, 4, 5], n),
    "TEHRUSLT": np.where(rng.random(n) < .05, -4, rng.integers(10, 60, n)),
    "TEIO1OCD": occ, "TUFNWGTP": rng.uniform(1e3, 1e4, n),
    "TU20FWGT": np.where(years >= 2020, rng.uniform(1e3, 1e4, n), -1),
})
k = rng.integers(3, 25, n)
ids = np.repeat(resp["TUCASEID"].to_numpy(), k)
act = pd.DataFrame({
    "TUCASEID": ids,
    "TRCODEP": rng.choice([10101, 50101, 50102, 120303, 110101], len(ids)),
    "TUACTDUR24": rng.integers(5, 120, len(ids)),
    "TEWHERE": rng.choice([1, 2, 3], len(ids)),
})
cps = pd.DataFrame({"TUCASEID": resp["TUCASEID"], "TULINENO": 1,
                    "GESTFIPS": rng.integers(1, 57, n)})
for stem, d in [("atusresp", resp), ("atusact", act), ("atuscps", cps)]:
    d.to_csv(raw / f"{stem}_0324.dat", index=False)
print("synthetic fixture written to", raw)
