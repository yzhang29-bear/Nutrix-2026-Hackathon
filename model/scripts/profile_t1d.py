"""Column profile of the T1D parquet without loading everything at once."""
import sys, json
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
f = sys.argv[1]
pf = pq.ParquetFile(f)
N = pf.metadata.num_rows
out = {}
for name in pf.schema_arrow.names:
    col = pq.read_table(f, columns=[name]).column(0)
    typ = str(col.type)
    nn = N - col.null_count
    info = dict(type=typ, non_null=nn, null_frac=round(col.null_count / N, 4))
    if pa.types.is_string(col.type) or pa.types.is_boolean(col.type):
        vc = pc.value_counts(col).to_pylist()
        vc = sorted(vc, key=lambda x: -x["counts"])
        info["n_unique"] = len(vc)
        info["top"] = [(str(x["values"]), x["counts"]) for x in vc[:25]]
    elif nn:
        c = pc.drop_null(col)
        if pa.types.is_timestamp(col.type):
            info["min"], info["max"] = str(pc.min(c).as_py()), str(pc.max(c).as_py())
        else:
            q = pc.quantile(c, q=[0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).to_pylist()
            info["q"] = [round(x, 3) for x in q]
            info["mean"] = round(pc.mean(c).as_py(), 3)
            info["n_unique_approx"] = pc.count_distinct(c).as_py() if nn < 5e7 else None
    out[name] = info
    print(name, json.dumps(info)[:600], flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
