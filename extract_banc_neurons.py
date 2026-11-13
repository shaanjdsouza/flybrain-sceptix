"""
extract_banc_neurons.py
-----------------------
Pull the neurons for the fly-brain -> MuJoCo walking-bot pipeline out of the
BANC v888 metadata table, and (optionally) build a small connected test circuit.

Pipeline groups (matches your diagram):
  ORN -> AL_PN / AL_LN -> KC (+DAN) -> MBON -> DN -> VNC premotor -> MN (leg)

USAGE
  1) Inspect first (always do this once):
       python extract_banc_neurons.py --meta banc_888_meta.parquet --inspect
  2) Extract all groups + a small test subset:
       python extract_banc_neurons.py --meta banc_888_meta.parquet --n-per-group 20
  3) Add connectivity (edgelist parquet/feather/csv from the BANC deposit):
       python extract_banc_neurons.py --meta banc_888_meta.parquet \
           --edges banc_888_edgelist.parquet --n-per-group 20

Outputs (in ./banc_subset/):
  all_groups.csv        every neuron matched, with its group label
  test_subset.csv       small subset for pipeline testing
  test_edges.csv        synapse-count edges inside the subset (if --edges given)
  group_summary.csv     counts per group

NOTE: I could not see your exact file, so column names / label strings are
auto-detected and the regex patterns below are the part to tune after --inspect.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# 1. Group definitions  (regex, case-insensitive, searched in label columns)
#    Tune these after running --inspect and looking at the real label strings.
# --------------------------------------------------------------------------
GROUPS = {
    # name: dict(patterns=[...], super_class=optional regex to restrict)
    "ORN": dict(patterns=[r"olfactory_receptor", r"\bORN\b"]),
    "AL_PN": dict(patterns=[r"antennal_lobe_projection", r"olfactory_projection",
                            r"multiglomerular_projection", r"uniglomerular_projection",
                            r"\bPN\b"]),
    "AL_LN": dict(patterns=[r"antennal_lobe_local", r"olfactory_local",
                            r"\bALLN\b", r"\bLN\b"]),
    "KC": dict(patterns=[r"kenyon", r"\bKC"]),
    "DAN": dict(patterns=[r"\bDAN\b", r"\bPPL1", r"\bPAM", r"dopaminergic"]),
    "MBON": dict(patterns=[r"\bMBON", r"mushroom_body_output"]),
    "DN": dict(patterns=[r"descending"], ),
    # motor neurons: restrict to those that look like leg MNs below
    "MN_leg": dict(patterns=[r"motor"], leg_filter=True),
}

# Strings that identify LEG motor neurons (vs wing / haltere / neck / abdomen)
LEG_PAT = r"leg|\bT[123]\b|front|middle|hind|prothoracic|mesothoracic|metathoracic|coxa|femur|tibia|tarsus|trochanter"
NON_LEG_PAT = r"wing|haltere|neck|abdom|salivary|pharyn|proboscis|crop"

# Candidate column names (first one found wins). Adjust if --inspect shows otherwise.
ID_CANDIDATES = ["root_id", "root_626", "root_888", "pt_root_id", "id", "bodyid", "segment_id"]
LABEL_COLS = ["flow", "super_class", "class", "sub_class", "primary_cell_type",
              "alternative_cell_types", "community_labels", "function", "body_part",
              "nerve", "hemilineage", "soma_side",
              "cell_class", "cell_sub_class", "cell_type"]
EDGE_PRE = ["pre", "pre_id", "pre_root_id", "source", "src", "from"]
EDGE_POST = ["post", "post_id", "post_root_id", "target", "dst", "to"]
EDGE_W = ["weight", "syn_count", "synapses", "n_syn", "count", "n"]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_any(path):
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith((".feather", ".arrow")):
        return pd.read_feather(path)
    if p.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if p.endswith((".csv.gz", ".gz")):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def first_present(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_cols(df):
    """'Root ID' -> 'root_id', 'Super Class' -> 'super_class', 'Sub Class' -> 'sub_class' ..."""
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns]
    return df


def inspect_meta(meta):
    print(f"\nRows: {len(meta):,}   Columns: {meta.shape[1]}\n")
    print("Columns:", list(meta.columns), "\n")
    for col in ["flow", "super_class", "class", "sub_class", "nerve", "body_part", "function"]:
        c = first_present(meta.columns, [col])
        if c:
            print(f"--- value counts: {c} ---")
            print(meta[c].value_counts(dropna=False).head(60).to_string(), "\n")
    print("Tip: search any keyword, e.g.  meta[meta.apply(lambda r: r.astype(str).str.contains('kenyon', case=False).any(), axis=1)]")


def build_label_text(meta, label_cols):
    """One lowercase string per neuron made of all its label columns."""
    present = [c for c in label_cols if c in meta.columns]
    text = pd.Series("", index=meta.index)
    for c in present:
        text = text + " | " + meta[c].fillna("").astype(str)
    return text, present


def match_group(text, spec):
    pat = "|".join(f"(?:{p})" for p in spec["patterns"])
    m = text.str.contains(pat, case=False, regex=True, na=False)
    if spec.get("leg_filter"):
        m &= text.str.contains(LEG_PAT, case=False, regex=True, na=False)
        m &= ~text.str.contains(NON_LEG_PAT, case=False, regex=True, na=False)
    return m


# --------------------------------------------------------------------------
# main extraction
# --------------------------------------------------------------------------
def extract_groups(meta, id_col):
    # Use only columns that exist; prefer the hierarchy columns for matching.
    text, used = build_label_text(meta, LABEL_COLS)
    print(f"Matching against label columns: {used}")

    parts = []
    for name, spec in GROUPS.items():
        mask = match_group(text, spec)

        # Guard: DN must really be descending, MN must be motor, etc.
        if "super_class" in meta.columns:
            sc = meta["super_class"].astype(str).str.lower()
            if name == "DN":
                mask &= sc.str.contains("descending")
            elif name == "MN_leg":
                mask &= sc.str.contains("motor")
            elif name in ("ORN",):
                mask &= sc.str.contains("sensory|olfactory") | mask  # keep, ORN is sensory
        sub = meta.loc[mask, [id_col] + [c for c in used if c != id_col]].copy()
        sub["group"] = name
        parts.append(sub)
        print(f"  {name:7s} {len(sub):>7,}")

    out = pd.concat(parts, ignore_index=True)
    # A neuron may match several groups (e.g. AL_PN vs AL_LN regex overlap).
    # Keep first match in GROUPS order, but report overlaps.
    dup = out[id_col].duplicated(keep=False).sum()
    if dup:
        print(f"  (!) {dup} rows matched >1 group; keeping first in GROUPS order. "
              f"Tighten the regexes if this is large.")
    out = out.drop_duplicates(id_col, keep="first").reset_index(drop=True)
    out["leg"] = ""
    mn = out.group == "MN_leg"
    sc = out.get("sub_class", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    side = out.get("soma_side", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    seg = pd.Series("", index=out.index)
    seg[sc.str.contains("front")] = "T1"
    seg[sc.str.contains("middle")] = "T2"
    seg[sc.str.contains("hind")] = "T3"
    lr = pd.Series("", index=out.index)
    lr[side.str.startswith("l")] = "L"
    lr[side.str.startswith("r")] = "R"
    out.loc[mn, "leg"] = (seg + "_" + lr)[mn]   # e.g. T1_L, T2_R, T3_L
    print("\n  Leg motor neurons per leg:")
    print(out.loc[mn, "leg"].value_counts().sort_index().to_string())
    return out


def load_edges(path):
    e = normalize_cols(read_any(path))
    pre = first_present(e.columns, EDGE_PRE)
    post = first_present(e.columns, EDGE_POST)
    w = first_present(e.columns, EDGE_W)
    if not (pre and post):
        raise ValueError(f"Could not detect pre/post columns in edgelist: {list(e.columns)}")
    if not w:
        e["_w"] = 1
        w = "_w"
    e = e[[pre, post, w]].rename(columns={pre: "pre", post: "post", w: "syn"})
    # collapse duplicates (per-neurotransmitter rows etc.)
    return e.groupby(["pre", "post"], as_index=False)["syn"].sum()


def pick_motor_neurons(groups_df, id_col, edges, mn_per_leg, min_syn):
    """Same number of leg motor neurons from each of the 6 legs (T1/T2/T3 x L/R).
    With edges: prefer MNs that receive the most synapses."""
    mn = groups_df[groups_df.group == "MN_leg"]
    indeg = None
    if edges is not None:
        indeg = edges[edges.syn >= min_syn].groupby("post").syn.sum()
    picked = []
    for leg, d in mn.groupby("leg"):
        ids = d[id_col]
        if indeg is not None:
            ids = ids.iloc[(-indeg.reindex(ids).fillna(0).values).argsort()]
        picked += list(ids.iloc[:mn_per_leg])
    return set(picked)


def build_test_subset(groups_df, id_col, edges, n_per_group, min_syn, mn_per_leg=5):
    """
    Small connected circuit.
      - If edges are given: walk the chain ORN->PN->KC->MBON->DN->MN and keep
        neurons that actually connect to the previous stage (strongest first).
      - Otherwise: random sample per group.
    """
    ids_by_group = {g: set(d[id_col]) for g, d in groups_df.groupby("group")}

    if edges is None:
        sub = pd.concat([d.sample(min(len(d), n_per_group), random_state=0)
                         for g, d in groups_df.groupby("group") if g != "MN_leg"],
                        ignore_index=True)
        mn_ids = pick_motor_neurons(groups_df, id_col, None, mn_per_leg, min_syn)
        sub = pd.concat([sub, groups_df[groups_df[id_col].isin(mn_ids)]], ignore_index=True)
        return sub, None

    # forward chain (order matters)
    chain = [("ORN", "AL_PN"), ("AL_PN", "KC"), ("KC", "MBON"),
             ("DAN", "MBON"), ("MBON", "DN"), ("DN", "MN_leg")]
    # DNs rarely hit leg MNs monosynaptically; allow one VNC premotor hop:
    e = edges[edges.syn >= min_syn]
    chosen = {g: set() for g in ids_by_group}

    # seed: top-N ORNs by outgoing synapses to PNs
    def top_targets(src_ids, tgt_ids, n):
        sel = e[e.pre.isin(src_ids) & e.post.isin(tgt_ids)]
        return set(sel.groupby("post").syn.sum().nlargest(n).index)

    def top_sources(src_ids, tgt_ids, n):
        sel = e[e.pre.isin(src_ids) & e.post.isin(tgt_ids)]
        return set(sel.groupby("pre").syn.sum().nlargest(n).index)

    # work backwards from motor neurons so the subset can actually reach legs
    chosen["MN_leg"] = pick_motor_neurons(groups_df, id_col, edges, mn_per_leg, min_syn)

    # VNC premotor: anything (not in our groups) that feeds chosen MNs strongly
    pre_motor_all = set(e[e.post.isin(chosen["MN_leg"])].groupby("pre").syn.sum()
                        .nlargest(n_per_group * 2).index) - chosen["MN_leg"]
    # DN pool: DNs that feed premotor/MN (wide pool, narrowed below)
    dn_pool = top_sources(ids_by_group.get("DN", set()),
                          pre_motor_all | chosen["MN_leg"], n_per_group * 5)

    # MBON -> (relay interneurons) -> DN. Direct MBON->DN links are rare in the fly,
    # so look for 1-hop relays that receive from MBONs and send to the DN pool.
    in_groups = set(groups_df[id_col])
    mb_ids = ids_by_group.get("MBON", set())
    from_mb = e[e.pre.isin(mb_ids)].groupby("post").syn.sum()
    to_dn = e[e.post.isin(dn_pool)].groupby("pre").syn.sum()
    both = from_mb.index.intersection(to_dn.index).difference(list(in_groups))
    relay_score = (from_mb[both] * to_dn[both]) ** 0.5
    chosen["relay"] = set(relay_score.nlargest(n_per_group).index)

    # DNs reached from MBONs (directly or via a relay)
    chosen["DN"] = top_targets(mb_ids | chosen["relay"], dn_pool, n_per_group)
    if not chosen["DN"]:
        print("(!) No DN is reachable from MBONs within 1 relay hop; using top DNs by motor output only.")
        chosen["DN"] = set(list(dn_pool)[:n_per_group])
    chosen["MBON"] = top_sources(mb_ids, chosen["relay"] | chosen["DN"], n_per_group)
    chosen["KC"] = top_sources(ids_by_group.get("KC", set()), chosen["MBON"], n_per_group)
    chosen["DAN"] = top_sources(ids_by_group.get("DAN", set()), chosen["MBON"], n_per_group)
    chosen["AL_PN"] = top_sources(ids_by_group.get("AL_PN", set()), chosen["KC"], n_per_group)
    chosen["ORN"] = top_sources(ids_by_group.get("ORN", set()), chosen["AL_PN"], n_per_group)
    chosen["AL_LN"] = top_sources(ids_by_group.get("AL_LN", set()),
                                  chosen["AL_PN"] | chosen["ORN"], n_per_group)

    keep = set().union(*chosen.values())
    sub = groups_df[groups_df[id_col].isin(keep)].copy()

    # relay + premotor layers (not in the label groups) so the sim has a path to the legs
    rl = pd.DataFrame({id_col: sorted(chosen["relay"] - set(sub[id_col]))})
    rl["group"] = "relay"
    pm = pd.DataFrame({id_col: sorted(pre_motor_all - keep)})
    pm["group"] = "VNC_premotor"
    sub = pd.concat([sub, rl, pm], ignore_index=True)

    empty = [g for g in ["ORN", "AL_PN", "KC", "DAN", "MBON", "DN", "MN_leg"]
             if g not in set(sub.group)]
    if empty:
        print(f"(!) Stages with no connected neurons in the subset: {empty}. "
              f"Try a lower --min-syn (e.g. 2) or a larger --n-per-group.")

    all_ids = set(sub[id_col])
    sub_edges = edges[edges.pre.isin(all_ids) & edges.post.isin(all_ids) & (edges.syn >= min_syn)]
    return sub, sub_edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="BANC v888 meta file (.parquet/.feather/.csv)")
    ap.add_argument("--edges", default=None, help="optional edgelist (.parquet/.feather/.csv)")
    ap.add_argument("--out", default="banc_subset")
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--mn-per-leg", type=int, default=5, help="leg motor neurons kept per leg (6 legs)")
    ap.add_argument("--min-syn", type=int, default=5, help="min synapses for an edge to count")
    ap.add_argument("--inspect", action="store_true", help="just print columns & label values")
    args = ap.parse_args()

    meta = normalize_cols(read_any(args.meta))
    if args.inspect:
        inspect_meta(meta)
        return

    id_col = first_present(meta.columns, ID_CANDIDATES)
    if id_col is None:
        raise SystemExit(f"No ID column found. Columns: {list(meta.columns)}. "
                         f"Edit ID_CANDIDATES at the top of the script.")
    print(f"Using ID column: {id_col}\n")

    groups_df = extract_groups(meta, id_col)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    groups_df.to_csv(out / "all_groups.csv", index=False)
    groups_df.group.value_counts().rename("n").to_csv(out / "group_summary.csv")

    edges = load_edges(args.edges) if args.edges else None
    if edges is not None:
        print(f"\nLoaded {len(edges):,} edges")
        meta_ids = set(groups_df[id_col]) | set(meta[id_col])
        overlap = edges.pre.isin(meta_ids).mean()
        print(f"Edge pre-IDs found in your neuron table: {overlap:.1%}")
        if overlap < 0.5:
            print("(!) Low overlap: this edgelist is probably from a different dataset/version "
                  "than neurons.csv (e.g. FlyWire FAFB vs BANC). Connectivity will be unreliable.")

    sub, sub_edges = build_test_subset(groups_df, id_col, edges, args.n_per_group, args.min_syn, args.mn_per_leg)
    sub.to_csv(out / "test_subset.csv", index=False)
    print(f"\nTest subset: {len(sub)} neurons")
    print(sub.group.value_counts().to_string())
    if sub_edges is not None:
        sub_edges.to_csv(out / "test_edges.csv", index=False)
        print(f"Test subset edges: {len(sub_edges):,}")
    print(f"\nWrote files to {out.resolve()}")


if __name__ == "__main__":
    main()