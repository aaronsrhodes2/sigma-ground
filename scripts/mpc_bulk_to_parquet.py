"""The MPC observation archive (MPCAT-OBS: NumObs 9.1 GB gz, UnnObs, CmtObs) to parquet, in parts, resumable: the whole-sky scorer's monthly source (stage 2, phase 1).

    python scripts/mpc_bulk_to_parquet.py [--batch 2026-10] [--files NumObs UnnObs CmtObs] [--lines-per-part 5000000] [--max-lines N]

Reads D:/datasets/mpc/bulk/<file>.txt.gz (fetched by scripts/fetch_mpc_bulk.sh) line by line, keeps the optical CCD/CMOS lines with their packed designation
(sigma_ground.field.interface.obs80), and writes D:/datasets/mpc_sky/obs/<batch>/<file>_part<k>.parquet every --lines-per-part input lines. A part that exists is skipped
(the gz is re-read up to it, since gzip cannot seek, but nothing is rewritten), so a cut night-window job resumes where it stopped. The archive is ordered by body, not by
publication, so a new monthly batch is a new <batch> directory and the scorer diffs it against the previous one by (packed, obstime, station). Progress in progress.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from sigma_ground.field.interface import obs80                        # noqa: E402

BULK = Path("D:/datasets/mpc/bulk")
OUT = Path("D:/datasets/mpc_sky/obs")
SCHEMA = pa.schema([("packed", pa.string()), ("packed_number", pa.string()), ("packed_prov", pa.string()), ("note2", pa.string()), ("year", pa.int16()), ("month", pa.int8()),
                    ("day", pa.float64()), ("ra_rad", pa.float64()), ("dec_rad", pa.float64()), ("mag", pa.float32()), ("band", pa.string()), ("station", pa.string())])


def convert(name: str, batch: str, lines_per_part: int, max_lines: int | None, log) -> dict:
    src = BULK / f"{name}.txt.gz"
    out = OUT / batch; out.mkdir(parents=True, exist_ok=True)
    prog_f = out / f"{name}_progress.json"
    prog = json.loads(prog_f.read_text(encoding="utf-8")) if prog_f.exists() else {"file": str(src), "parts": {}, "lines_in": 0, "rows_out": 0}
    t0 = time.time(); part = 0; rows = []; n_in = 0; n_out = 0
    with gzip.open(src, "rt", encoding="latin-1") as f:
        for ln in f:
            n_in += 1
            if (n_in - 1) // lines_per_part != part:                     # a part boundary
                _flush(rows, out, name, part, prog, n_in - 1, n_out, prog_f, log); rows = []; part += 1
            if str(part) in prog["parts"]:
                continue                                                 # already written: read past it
            r = obs80.parse_line(ln)
            if r is not None:
                rows.append(r); n_out += 1
            if max_lines and n_in >= max_lines:
                break
    _flush(rows, out, name, part, prog, n_in, n_out, prog_f, log)
    prog["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds"); prog["seconds"] = round(time.time() - t0)
    prog_f.write_text(json.dumps(prog, indent=1), encoding="utf-8")
    log(f"{name}: {n_in:,} lines read, {n_out:,} optical rows written this run, {time.time() - t0:.0f} s")
    return prog


def _flush(rows, out: Path, name: str, part: int, prog: dict, lines_in: int, rows_out: int, prog_f: Path, log):
    if str(part) in prog["parts"]:
        return
    f = out / f"{name}_part{part:04d}.parquet"
    tab = pa.Table.from_pydict(obs80.rows_to_columns(rows), schema=SCHEMA) if rows else pa.Table.from_pydict({c: [] for c in obs80.COLUMNS}, schema=SCHEMA)
    tmp = f.with_suffix(".tmp.parquet"); pq.write_table(tab, tmp, compression="zstd"); tmp.replace(f)
    prog["parts"][str(part)] = {"rows": tab.num_rows, "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    prog["lines_in"] = lines_in; prog["rows_out"] = rows_out
    prog_f.write_text(json.dumps(prog, indent=1), encoding="utf-8")
    log(f"  {f.name}: {tab.num_rows:,} rows (lines read so far {lines_in:,})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default=None, help="the archive batch label (default: the bulk files' modification month)")
    ap.add_argument("--files", nargs="+", default=["CmtObs", "UnnObs", "NumObs"])
    ap.add_argument("--lines-per-part", type=int, default=5_000_000)
    ap.add_argument("--max-lines", type=int, default=None, help="stop after this many input lines per file (a timing run)")
    a = ap.parse_args()
    log = lambda s: print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {s}", flush=True)
    batch = a.batch or datetime.fromtimestamp((BULK / "NumObs.txt.gz").stat().st_mtime, tz=timezone.utc).strftime("%Y-%m")
    for name in a.files:
        convert(name, batch, a.lines_per_part, a.max_lines, log)
    log("finished: bulk -> parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
