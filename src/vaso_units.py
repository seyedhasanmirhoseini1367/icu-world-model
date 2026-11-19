"""
vaso_units.py - Stage 11, step 4 (report only): which units do the vasopressor rows use?

The extraction (mimic_data.extract) converts a vasopressor rate to norepinephrine-equivalent
mcg/kg/min with
    rate * NEE_FACTOR[itemid] * (CASE rateuom WHEN 'mcg/min'    THEN 1 / weight
                                             WHEN 'units/hour' THEN 1 / 60
                                             ELSE 1.0 END)
Rows whose unit is neither 'mcg/min' nor 'units/hour' fall into the "ELSE 1.0" branch, which is
only right if the unit is already per kg per minute. This counts, per itemid, the rows per
rateuom and how many of them take that branch. It uses the same row filters as the extraction
(rate > 0, not 'Rewritten'), once over all rows and once only for rows of the extracted stays.
It changes nothing in the extraction. Only counts are returned.
"""
import duckdb

from mimic_data import MIN_HOURS, NEE_FACTOR, find_files, src

ITEM_NAMES = {221906: "norepinephrine", 221289: "epinephrine", 229617: "epinephrine (229617)",
              221749: "phenylephrine", 221662: "dopamine", 222315: "vasopressin"}
EXPLICIT_UOMS = ("mcg/min", "units/hour")


def count_units(root):
    con = duckdb.connect()
    in_cols = dict(stay_id="BIGINT", itemid="BIGINT", rate="DOUBLE", rateuom="VARCHAR",
                   statusdescription="VARCHAR")
    stay_cols = dict(stay_id="BIGINT", intime="TIMESTAMP", outtime="TIMESTAMP")
    items = ", ".join(map(str, NEE_FACTOR))
    rows = con.execute(f"""
        WITH kept AS (
            SELECT stay_id FROM {src(find_files(root, 'icustays'), stay_cols)}
            WHERE date_diff('second', intime, outtime) >= {MIN_HOURS} * 3600),
        v AS (
            SELECT i.itemid, coalesce(i.rateuom, '<NULL>') AS rateuom,
                   (i.stay_id IN (SELECT stay_id FROM kept)) AS in_stays
            FROM {src(find_files(root, 'inputevents'), in_cols)} i
            WHERE i.itemid IN ({items}) AND i.rate > 0
              AND coalesce(i.statusdescription, '') <> 'Rewritten')
        SELECT itemid, rateuom, count(*) AS n_rows, sum(in_stays::INT) AS n_rows_in_stays
        FROM v GROUP BY itemid, rateuom ORDER BY itemid, n_rows DESC""").fetchall()
    out = {}
    for itemid, uom, n, n_in in rows:
        d = out.setdefault(str(itemid), dict(name=ITEM_NAMES.get(itemid, "?"), factor=NEE_FACTOR[itemid],
                                             by_rateuom={}, else_branch_rows=0,
                                             else_branch_rows_in_stays=0, total_rows=0))
        d["by_rateuom"][uom] = dict(n_rows=int(n), n_rows_in_stays=int(n_in or 0),
                                    else_branch=uom not in EXPLICIT_UOMS)
        d["total_rows"] += int(n)
        if uom not in EXPLICIT_UOMS:
            d["else_branch_rows"] += int(n)
            d["else_branch_rows_in_stays"] += int(n_in or 0)
    return out


def report(r):
    lines = [f"{'itemid':>7} {'name':<22} {'rateuom':<14} {'rows':>10} {'in stays':>10}  ELSE 1.0?"]
    for itemid, d in r.items():
        for uom, c in d["by_rateuom"].items():
            lines.append(f"{itemid:>7} {d['name']:<22} {uom:<14} {c['n_rows']:>10,} {c['n_rows_in_stays']:>10,}  "
                         f"{'yes' if c['else_branch'] else 'no'}")
        lines.append(f"{'':>7} {'':<22} {'-> ELSE 1.0 rows':<14} {d['else_branch_rows']:>10,} "
                     f"{d['else_branch_rows_in_stays']:>10,}")
    return "\n".join(lines)
