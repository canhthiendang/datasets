# ATUS pipeline (untested on real data)

Scripts to build a person-day file from the BLS American Time Use Survey and
estimate whether the gap between reported usual hours and diary-measured work
hours changed after 2020 in occupations exposed to remote work.

Status: runs end-to-end on synthetic fixtures only (`tests/make_fixture.py`).
No real estimates have been produced; the build environment could not reach
bls.gov. Variable names follow the BLS codebook and should be checked against
the current release before use.

```
pip install pandas numpy pyfixest pyarrow
python build_atus.py --raw-dir data/raw --out data/atus_personday.parquet --download
python analysis_hours_gap.py --data data/atus_personday.parquet --out-dir output
```
