# ---------------- CONFIG ----------------
GLOBAL = ["Pac", "Acc", "Det"]  # stays global

# Position profiles: list attributes by importance group
PROFILES = {
    "GK": {
        "primary":   ["Ref", "1v1", "Aer", "Pos", "Dec"],
        "secondary": ["Cmd", "Han", "Kic", "Cnt", "Agi"],
        "tertiary":  ["Com", "Thr", "Ant"],
    },
    "FB": {
        "primary":   ["Wor", "Sta", "Pos", "Cro", "Mar"],
        "secondary": ["OtB", "Tec", "Ant", "Tea"],
        "tertiary":  ["Tck", "Dri", "Fir", "Pas", "Cnt", "Dec", "Agi", "Bal"],
    },
    "CB": {
        "primary":   ["Jum", "Tck", "Cmp", "Cnt", "Str", "Ant", "Pos"],
        "secondary": ["Agg", "Bra", "Dec"],
        "tertiary":  ["Hea"],
    },
    "DM": {
        "primary":   ["Fir", "Tea", "Cmp", "Pos", "Pas", "Tck", "Ant"],
        "secondary": ["Tec", "Dec", "Vis"],
        "tertiary":  ["Cnt", "Mar", "Agg", "Wor", "Sta", "Str", "Bal"],
    },
    "CM": {
        "primary":   ["Wor", "Sta", "Pas", "Dec", "Fir"],
        "secondary": ["Lon", "Tck", "OtB", "Tea", "Cmp", "Vis", "Tec"],
        "tertiary":  ["Dri", "Fin", "Agg", "Ant", "Pos", "Bal", "Str", "Agi"],
    },
    "W": {
        "primary":   ["Agi", "Dri", "Bal", "Fla"],
        "secondary": ["Fin", "Fir", "OtB", "Cro", "Tec"],
        "tertiary":  ["Lon", "Pas", "Wor", "Sta", "Ant", "Cmp"],
    },
    "AM": {
        "primary":   ["Vis", "Fla", "Fir", "OtB", "Tec", "Dec", "Pas"],
        "secondary": ["Lon", "Dri", "Ant", "Tea"],
        "tertiary":  ["Fin", "Cmp", "Agi"],
    },
    "ST": {
        "primary":   ["OtB", "Cmp", "Det", "Fir", "Fin", "Ant", "Jum"],
        "secondary": ["Tec", "Pas", "Dri"],
        "tertiary":  ["Wor", "Agi", "Bal", "Sta", "Lon", "Vis", "Tea", "Dec", "Hea", "Str"],
    },
}

# Blend of position profile vs global
ALPHA_BY_POS = {"GK": 0.98, "CB": 0.96, "FB": 0.88, "DM": 0.95,
                "CM": 0.93, "W": 0.87, "AM": 0.90, "ST": 0.88}

GROUP_GAMMA = {"primary": 2.2, "secondary": 1.7, "tertiary": 1.4}
GLOB_GAMMA  = 1.9

SCALE_LO, SCALE_HI = 42.0, 99.0


POS_CATEGORY = {
    "GK": "GK",
    "FB": "DEF",
    "CB": "DEF",
    "DM": "MID",
    "CM": "MID",
    "AM": "FWD",
    "W":  "FWD",
    "ST": "FWD",
}

# inclusive regex per category (matches if ANY token is present)
CATEGORY_PATTERNS = {
    "GK":  r"\bGK\b",
    "DEF": r"\bD\b|\bWB\b",
    # MID: DM or central M (not side-only M R/L/RL/LR)
    "MID": r"\bDM\b|\bM\b(?!\s*\((?:R|L|RL|LR)\))",
    # FWD: ST/W + wide AM + wide M treated as wingers
    "FWD": r"\bST\b|\bW\b|\bAM\b\s*\((?:R|L|RL|LR)\)|\bM\b\s*\((?:R|L|RL|LR)\)",
}


# Raw scores that should evaluate to ~90 after scaling (consistent across squads)
ANCHOR_90 = {
    "GK": 82,   # Alisson-tier raw
    "CB": 82,   # VVD-tier raw
    "FB": 82,
    "DM": 82,
    "CM": 82,
    "W":  82,   # Salah-tier raw on wing/AM/ST
    "AM": 82,
    "ST": 82,
}





# PosCalc.py — accepts .rtf or .txt FM exports
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from html import escape

# -------- RTF -> plain text (keeps your '|' table) --------
def _decode_rtf_hex(s: str) -> str:
    # turns \'hh into the corresponding character (cp1252)
    def repl(m):
        b = int(m.group(1), 16)
        return bytes([b]).decode("cp1252", errors="ignore")
    return re.sub(r"\\'([0-9a-fA-F]{2})", repl, s)

def rtf_to_text(raw: str) -> str:
    s = _decode_rtf_hex(raw)
    # line breaks and tabs
    s = s.replace("\\par", "\n")
    s = re.sub(r"\\line\b", "\n", s)
    s = re.sub(r"\\tab\b", "\t", s)
    # strip other control words (leave literal '|' alone)
    s = re.sub(r"\\[A-Za-z]+-?\d* ?", "", s)
    # drop group braces
    s = s.replace("{", "").replace("}", "")
    # collapse excessive blank lines
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

# -------- Reader for .rtf or .txt --------
def read_pipe_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".rtf":
        # read bytes to preserve hex escapes, then decode and clean
        raw = path.read_bytes().decode("latin-1", errors="ignore")
        text = rtf_to_text(raw)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    lines = text.splitlines()
    header, rows = None, []
    for ln in lines:
        s = ln.strip()
        if not s.startswith("|"):
            continue
        # skip separator rows like | ---- |
        if set(s.strip("|").strip()) == {"-"}:
            continue
        cells = [c.strip() for c in s.split("|")]
        if cells and cells[0] == "": cells = cells[1:]
        if cells and cells[-1] == "": cells = cells[:-1]
        if header is None:
            header = cells
        elif len(cells) == len(header):
            rows.append(dict(zip(header, cells)))
    if header is None:
        raise ValueError("No header row found. Ensure the RTF contains the pipe table (| ... |).")

    df = pd.DataFrame(rows)

    # Age (if present) may be like "31" or "31 (yc)"; extract digits
    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"].str.extract(r"(\d+)", expand=False), errors="coerce")

    # DO NOT coerce these to numeric (they're text in scouting lists)
    NON_NUMERIC = {"Name", "Position", "Age", "Rec", "Inf"}

    # Convert all other columns to numeric
    for col in df.columns:
        if col in NON_NUMERIC:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# --------------- SCORING ---------------
def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(values)
    w = weights * mask
    num = np.nansum(values * w, axis=1)
    den = np.sum(w, axis=1)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)

def _build_pos_matrix(df: pd.DataFrame, pos: str):
    prof = PROFILES[pos]
    prim = [c for c in prof["primary"]   if c in df.columns]
    sec  = [c for c in prof["secondary"] if c in df.columns and c not in prim]
    ter  = [c for c in prof["tertiary"]  if c in df.columns and c not in prim and c not in sec]
    # weights by group
    gw = prof.get("group_weights", {"primary":1.0, "secondary":0.5, "tertiary":0.25})
    w_prim = np.array([gw["primary"]]  * len(prim), dtype=float)
    w_sec  = np.array([gw["secondary"]]* len(sec),  dtype=float)
    w_ter  = np.array([gw["tertiary"]] * len(ter),  dtype=float)
    cols = prim + sec + ter
    weights = np.concatenate([w_prim, w_sec, w_ter]) if cols else np.array([], dtype=float)
    # apply per-attribute overrides
    overrides = prof.get("attr_weights", {})
    if overrides and cols:
        weights = np.array([weights[i] * float(overrides.get(cols[i], 1.0)) for i in range(len(cols))], dtype=float)
    return cols, weights

def _absolute_rescale_col(col, exp, lo=SCALE_LO, hi=SCALE_HI):
    x = pd.to_numeric(col, errors="coerce") / 100.0
    x = np.clip(x, 0, 1)
    x = np.power(x, exp)
    return lo + (hi - lo) * x

# --- one absolute scaler used for all positions (gentle) ---
SCALE_LO, SCALE_HI = 0, 99.0     # FIFA-like bounds
RAW_AT_90 = 82.0                    # raw 0–100 that should map to ~90
STRETCH   = 1.00                    # no extra widening (bump later to 1.05 if needed)
OFFSET    = -1.0                    # tiny downward nudge to avoid 95–99 spam

# ---- Absolute, position-agnostic mapping: raw(0–100) -> FIFA-like 46–99 with more spread
RAW_KNOTS  = np.array([ 0, 45, 55, 65, 75,  82,  88,  95, 100], dtype=float)
FIFA_KNOTS = np.array([46, 55, 65, 74, 82,  90,  94,  97,  99], dtype=float)
def _piecewise_rescale_col(col: pd.Series) -> pd.Series:
    x = pd.to_numeric(col, errors="coerce").clip(0, 100)
    y = np.interp(x, RAW_KNOTS, FIFA_KNOTS)   # consistent across squads
    return pd.Series(y, index=col.index)

def _exp_for_anchor(raw_at_90, lo=46.0, hi=99.0, target=90.0):
    p = (target - lo) / (hi - lo)
    return float(np.log(p) / np.log(raw_at_90 / 100.0))

EXP = _exp_for_anchor(RAW_AT_90, lo=SCALE_LO, hi=SCALE_HI, target=90.0)

def _final_scale(col):
    x = pd.to_numeric(col, errors="coerce") / 100.0
    x = x.clip(0.0, 1.0)
    x = np.power(x, EXP)                       # one exponent for all
    x = SCALE_LO + (SCALE_HI - SCALE_LO) * x   # map to [46, 99]
    mid = (SCALE_LO + SCALE_HI) / 2.0          # 72.5
    x = (x - mid) * STRETCH + mid + OFFSET     # gentle, global
    return np.clip(x, SCALE_LO, SCALE_HI)


    pos_cols, pos_w = _build_pos_matrix(df_pos, pos)
    pos_mean = _weighted_mean(df_pos[pos_cols].to_numpy(float), pos_w) if len(pos_cols) else np.full(len(df_pos), np.nan)

    # globals (de-dupe against pos columns)
    glob_cols = [g for g in GLOBAL if g in df_pos.columns and g not in pos_cols]
    glob_mean = np.nanmean(df_pos[glob_cols].to_numpy(float), axis=1) if glob_cols else np.full(len(df_pos), np.nan)

    alpha = ALPHA_BY_POS.get(pos, 0.9)

    both = ~np.isnan(pos_mean) & ~np.isnan(glob_mean)
    only_pos = ~np.isnan(pos_mean) & np.isnan(glob_mean)
    only_glob = np.isnan(pos_mean) & ~np.isnan(glob_mean)

    blend = np.full(len(df_pos), np.nan, dtype=float)
    blend[both]     = alpha * pos_mean[both] + (1 - alpha) * glob_mean[both]
    blend[only_pos] = pos_mean[only_pos]
    blend[only_glob]= glob_mean[only_glob]

    # scale 1–20 -> 0–100 and write back only for GK rows (or all rows for outfield)
    scaled = (blend / 20.0) * 100.0
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[df_pos.index] = scaled
    return out

# Category mapping used only for the report lists (doesn't change your scoring)
POS_TO_CAT = {"GK": "GK", "CB": "DEF", "FB": "DEF", "DM": "MID", "CM": "MID", "AM": "MID", "W": "FWD", "ST": "FWD"}

# ---- EXACT role parser (replace your _parse_roles / _plays_target / eligible_mask trio with this) ----
TOKENS = {"GK","D","WB","DM","M","AM","W","ST"}

def _parse_roles(text: str):
    """
    Expand 'M/AM (RLC)' -> [('M', {'R','L','C'}), ('AM', {'R','L','C'})]
    """
    roles = []
    s = str(text).upper().strip()
    if not s:
        return roles
    for seg in re.split(r"\s*,\s*", s):            # split by commas
        if not seg:
            continue
        m = re.search(r"\(([^)]*)\)", seg)         # grab letters in (...)
        letters = set()
        if m:
            letters = set(re.sub(r"[^RLC]", "", m.group(1)))
            seg = (seg[:m.start()] + seg[m.end():]).strip()
        for base in re.split(r"\s*/\s*", seg):     # split D/WB/M into tokens
            base = base.strip()
            if base in TOKENS:
                roles.append((base, letters.copy()))
    return roles

def _plays_target(roles, target: str) -> bool:
    RL = {"R","L"}
    if target == "GK":
        return any(b == "GK" for b,_ in roles)
    if target == "CB":
        # Only D with explicit central letter (C present)
        return any(b == "D" and ("C" in letters) for b,letters in roles)
    if target == "FB":
        # D or WB with R/L (full-backs/wing-backs only)
        return any((b in {"D","WB"}) and (letters & RL) for b,letters in roles)
    if target == "DM":
        return any(b == "DM" for b,_ in roles)
    if target == "CM":
        # M with central letter (C). If M has no letters at all, treat as central.
        return any(b == "M" and (("C" in letters) or (len(letters) == 0)) for b,letters in roles)
    if target == "AM":
        # AMC only (central); *not* AML/AMR
        return any(b == "AM" and (("C" in letters) or (len(letters) == 0)) for b,letters in roles)
    if target == "W":
        # True wingers (W) or wide AM/M (R/L)
        return any(
            b == "W" or ((b in {"AM","M"}) and (letters & RL))
            for b,letters in roles
        )
    if target == "ST":
        return any(b == "ST" for b,_ in roles)
    return False

def eligible_mask(obj, pos: str) -> pd.Series:
    """
    True only when the player can play the specific target position, based on
    their literal Position string. Works for a DataFrame or a Series.
    """
    if isinstance(obj, pd.Series):
        s = obj
        idx = obj.index
    else:
        s = obj["Position"]
        idx = obj.index
    return s.apply(lambda x: _plays_target(_parse_roles(x), pos)).reindex(idx, fill_value=False)

# --- Side-aware checks for full-backs (RB/LB) using your existing _parse_roles ---
def _is_fb_side(pos_text: str, side: str) -> bool:
    """True if player can play full-back on given side ('R' or 'L')."""
    roles = _parse_roles(pos_text)
    return any(b in ("D", "WB") and (side in letters) for b, letters in roles)

def _sort_block(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Sort by rating desc, then Age asc (younger first), then name."""
    tmp = sub.copy()
    tmp["__age_sort"] = pd.to_numeric(tmp.get("Age"), errors="coerce").fillna(10**6)
    return tmp.sort_values([col, "__age_sort", "Name"], ascending=[False, True, True]).drop(columns="__age_sort")

def build_best_xi(scores: pd.DataFrame) -> pd.DataFrame:
    """
    Pick a 4-3-3 Best XI that maximizes TOTAL rating:
      GK, RB, LB, CB1, CB2, DM, CM, AM, W1, W2, ST
    Each player can fill at most one slot. Ties broken by younger age.
    """
    # ---- slot definitions and eligibility masks ----
    def fb_side_mask(side: str) -> pd.Series:
        return eligible_mask(scores, "FB") & scores["Position"].apply(lambda t: _is_fb_side(t, side))

    slot_defs = [
        ("GK",  "GK",  eligible_mask(scores, "GK")),
        ("RB",  "FB",  fb_side_mask("R")),
        ("LB",  "FB",  fb_side_mask("L")),
        ("CB1", "CB",  eligible_mask(scores, "CB")),
        ("CB2", "CB",  eligible_mask(scores, "CB")),
        ("DM",  "DM",  eligible_mask(scores, "DM")),
        ("CM",  "CM",  eligible_mask(scores, "CM")),
        ("AM",  "AM",  eligible_mask(scores, "AM")),
        ("W1",  "W",   eligible_mask(scores, "W")),
        ("W2",  "W",   eligible_mask(scores, "W")),
        ("ST",  "ST",  eligible_mask(scores, "ST")),
    ]

    # Age for tie-break: younger is (slightly) better
    age = pd.to_numeric(scores.get("Age"), errors="coerce").fillna(40)
    EPS = 1e-4  # tiny so it only breaks ties of equal integer ratings

    # Build candidate lists per slot (player index, raw rating, adjusted score)
    cands = []
    slot_max = []
    for slot, col, mask in slot_defs:
        rr = pd.to_numeric(scores[col], errors="coerce")
        li = []
        for i in scores.index[mask]:
            r = rr.loc[i]
            if pd.isna(r):  # no rating for that slot
                continue
            adj = float(r) + EPS * (40 - float(age.loc[i]))  # younger -> larger adj
            li.append((int(i), float(r), adj))
        li.sort(key=lambda t: t[2], reverse=True)
        cands.append(li)
        slot_max.append(max((t[2] for t in li), default=-1e9))

    # If any slot has no candidates at all -> return empty (or fall back)
    if any(len(li) == 0 for li in cands):
        return pd.DataFrame(columns=["Slot", "Player", "Position", "Age", "Pos", "Rating"])

    # Order slots by fewest choices first (branching reduction)
    order = sorted(range(len(slot_defs)), key=lambda k: len(cands[k]))
    ordered_max = [slot_max[k] for k in order]
    # Upper-bound for pruning: suffix sums of best-possible remaining scores
    suf = [0.0] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suf[i] = suf[i + 1] + max(0.0, ordered_max[i])

    best_total = -1e18
    best_assign = [-1] * len(slot_defs)
    used = set()
    assign = [-1] * len(slot_defs)

    def dfs(i: int, total: float):
        nonlocal best_total, best_assign
        if i == len(order):
            if total > best_total:
                best_total = total
                best_assign = assign[:]
            return
        # prune if even the optimistic bound can't beat best_total
        if total + suf[i] <= best_total + 1e-12:
            return

        slot_idx = order[i]
        for pid, raw, adj in cands[slot_idx]:
            if pid in used:
                continue
            used.add(pid)
            assign[slot_idx] = pid
            dfs(i + 1, total + adj)
            assign[slot_idx] = -1
            used.remove(pid)

    dfs(0, 0.0)

    # Build the resulting table
    rows = []
    slot_order = ["GK", "RB", "CB1", "CB2", "LB", "DM", "CM", "AM", "W1", "W2", "ST"]
    slot_to_idx = {name: i for i, (name, _, _) in enumerate(slot_defs)}

    for slot_name in slot_order:
        si = slot_to_idx[slot_name]
        pid = best_assign[si]
        if pid == -1:
            continue
        col = slot_defs[si][1]
        r = scores.loc[pid]
        rows.append({
            "Slot":   slot_name,
            "Player": r["Name"],
            "Position": r["Position"],
            "Age":    (int(r["Age"]) if pd.notna(r.get("Age")) else ""),
            "Pos":    col,
            "Rating": int(pd.to_numeric(r[col], errors="coerce")),
        })

    return pd.DataFrame(rows, columns=["Slot", "Player", "Position", "Age", "Pos", "Rating"])

def pick_best_subs(scores: pd.DataFrame, best11: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """
    Return the top `n` substitutes by their BEST outfield position rating,
    excluding anyone already in best11 and excluding goalkeepers.
    Tie-break: younger age preferred.
    """
    used_names = set(best11["Player"])  # fine for a single-squad sheet
    field_cols = ["FB", "CB", "DM", "CM", "AM", "W", "ST"]

    age = pd.to_numeric(scores.get("Age"), errors="coerce").fillna(40)
    EPS = 1e-4

    rows = []
    for i, r in scores.iterrows():
        if r["Name"] in used_names:
            continue

        vals = pd.to_numeric(r[field_cols], errors="coerce")
        if vals.notna().sum() == 0:
            # no outfield ratings -> skip (pure GK or no eligible roles)
            continue

        best_pos = vals.idxmax()
        best_raw = float(vals.max())
        adj = best_raw + EPS * (40 - float(age.loc[i]))  # younger nudges up in ties

        rows.append({
            "_adj":   adj,                          # for sorting only
            "Player": r["Name"],
            "Position": r["Position"],
            "Age":    (int(age.loc[i]) if pd.notna(age.loc[i]) else ""),
            "Pos":    best_pos,
            "Rating": int(best_raw),
        })

    subs = pd.DataFrame(rows).sort_values("_adj", ascending=False).head(n).drop(columns=["_adj"])
    return subs.reset_index(drop=True)

def score_position(df: pd.DataFrame, pos: str) -> pd.Series:
    """Compute a raw 0–100 score for one position, aligned to df.index.
       Depends on: GLOBAL, ALPHA_BY_POS, eligible_mask, _build_pos_matrix, _weighted_mean."""
    # Keep only players eligible for this position/category
    df_pos = df[eligible_mask(df, pos)].copy()
    if df_pos.empty:
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Position-weighted mean
    pos_cols, pos_w = _build_pos_matrix(df_pos, pos)
    if len(pos_cols):
        pos_vals = df_pos[pos_cols].to_numpy(float)
        pos_mean = _weighted_mean(pos_vals, pos_w)
    else:
        pos_mean = np.full(len(df_pos), np.nan, dtype=float)

    # globals (de-dupe against pos columns) — ignore for GK
    if pos == "GK":
        glob_cols = []
        glob_mean = np.full(len(df_pos), np.nan)
    else:
        glob_cols = [g for g in GLOBAL if g in df_pos.columns and g not in pos_cols]
        glob_mean = np.nanmean(df_pos[glob_cols].to_numpy(float), axis=1) if glob_cols else np.full(len(df_pos), np.nan)

    alpha = ALPHA_BY_POS.get(pos, 0.9)

    both      = ~np.isnan(pos_mean) & ~np.isnan(glob_mean)
    only_pos  = ~np.isnan(pos_mean) &  np.isnan(glob_mean)
    only_glob =  np.isnan(pos_mean) & ~np.isnan(glob_mean)

    blend = np.full(len(df_pos), np.nan, dtype=float)
    blend[both]      = alpha * pos_mean[both] + (1 - alpha) * glob_mean[both]
    blend[only_pos]  = pos_mean[only_pos]
    blend[only_glob] = glob_mean[only_glob]

    # Convert 1–20 attribute space to 0–100 raw score
    raw = (blend / 20.0) * 100.0

    # Write back into a Series aligned to the full dataframe
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[df_pos.index] = raw
    return out

def generate_html_report(scores: pd.DataFrame, out_html: Path, title: str, best11: pd.DataFrame, subs: pd.DataFrame | None = None):
    """Lightweight, dependency-free HTML report with nice tables."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    pos_cols = [p for p in PROFILES.keys() if p in scores.columns]
    safe = scores.copy()

    # We'll show blanks for NaNs so tables look clean
    for c in pos_cols:
        safe[c] = safe[c].astype("Int64").astype("object").where(~safe[c].isna(), "")

    
# NEW: format Age column if present
    if "Age" in safe.columns:
        safe["Age"] = (pd.to_numeric(safe["Age"], errors="coerce")
                     .astype("Int64")
                     .astype("object")
                     .where(~safe["Age"].isna(), ""))

    # Build Top-10 lists per position, filtered by eligibility
    sections = []

        # === NEW: Top 6 — Overall (best position per player) ===
    pos_cols = [p for p in PROFILES.keys() if p in scores.columns]
    id_cols  = [c for c in ["Name", "Position", "Age"] if c in scores.columns]

    long = scores.melt(
        id_vars=id_cols,
        value_vars=pos_cols,
        var_name="RatedPos",
        value_name="Rating"
    ).dropna(subset=["Rating"])

    # Best position for each player (highest rating)
    idx   = long.groupby("Name")["Rating"].idxmax()
    best  = long.loc[idx].copy()
    best["Rating"] = best["Rating"].round().astype(int)

    # Top 6 overall
    top6 = best.sort_values("Rating", ascending=False).head(6)

    # Pretty columns (match other sections: Player, Position, Age, Pos, Rating)
    display_cols = (["Name", "Position", "Age", "RatedPos", "Rating"]
                    if "Age" in scores.columns
                    else ["Name", "Position", "RatedPos", "Rating"])
    top6_disp = top6[display_cols].rename(columns={"Name":"Player", "RatedPos":"Pos"})

    # Make Age look clean (blank if missing)
    if "Age" in top6_disp.columns:
        top6_disp["Age"] = (pd.to_numeric(top6_disp["Age"], errors="coerce")
                               .astype("Int64")
                               .astype("object")
                               .where(~top6_disp["Age"].isna(), ""))

    sections.append(f"""
      <section>
        <h2>Top 6 — Overall</h2>
        {top6_disp.to_html(index=False, classes="grid")}
      </section>
    """)

    # --- Best XI + Bench side-by-side (at the very top of sections) ---
    bx_html = ""
    sb_html = ""

    if best11 is not None and not best11.empty:
        bx_html = best11.rename(columns={"Pos": "Role"}).to_html(index=False, classes="grid")

    if subs is not None and not subs.empty:
        sh = subs.copy().rename(columns={"Pos": "Best Pos"})
        sb_html = sh.to_html(index=False, classes="grid")

    if bx_html or sb_html:
        sections.insert(0, f"""
      <div class="row2">
        {'<section><h2>Best XI — 4-3-3</h2>' + bx_html + '</section>' if bx_html else ''}
        {'<section><h2>Bench — Best 9 (any position, no GK)</h2>' + sb_html + '</section>' if sb_html else ''}
      </div>
    """)

    for pos in pos_cols:
        cols_for_top = (["Name", "Position", "Age", pos]
                if "Age" in safe.columns else
                ["Name", "Position", pos])
        sub = safe[cols_for_top].dropna(subset=[pos]).copy()
        sub = sub[eligible_mask(sub["Position"], pos)]
        top = sub.sort_values(pos, ascending=False).head(6).copy()
        top[pos] = top[pos].astype(int)
        if top.empty:
            continue
        sections.append(
            f"""
            <section>
              <h2>{pos} — Top 6</h2>
              {top.rename(columns={"Name": "Player"}).to_html(index=False, classes="grid")}
            </section>
            """
        )

    # Full squad grid
    cols_for_top = (["Name", "Position", "Age", pos]
                if "Age" in safe.columns else
                ["Name", "Position", pos])
    sub = safe[cols_for_top].dropna(subset=[pos]).copy()
    pos_cols  = [p for p in PROFILES.keys() if p in scores.columns]
    base_cols = ["Name", "Position"] + (["Age"] if "Age" in safe.columns else [])
    squad = safe[base_cols + pos_cols].copy()
    squad = squad.rename(columns={"Name": "Player"})
    squad_html = squad.to_html(index=False, classes="grid wide")

    css = """
    <style>
    .row2 { display:grid; gap:16px; }
    @media (min-width: 900px) { .row2 { grid-template-columns: 1fr 1fr; } }
    .row2 > section { margin:0; } /* use existing section card styling */

      :root { --bg:#0f1115; --card:#171a21; --text:#e8eaed; --sub:#9aa0a6; --accent:#5ee; }
      * { box-sizing:border-box; }
      body { margin:32px; background:var(--bg); color:var(--text); font:14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; }
      h1 { font-size:28px; margin:0 0 6px; }
      h2 { font-size:20px; margin:24px 0 8px; }
      .meta { color:var(--sub); margin-bottom:18px; }
      section { background:var(--card); padding:16px; border-radius:14px; box-shadow:0 6px 20px rgba(0,0,0,.25); margin-bottom:16px; }
      .grid { width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border-radius:10px; }
      .grid thead th { background:#20242f; color:#cbd5e1; text-align:left; padding:10px 12px; position:sticky; top:0; }
      .grid tbody td { padding:10px 12px; border-top:1px solid #2a2f3c; }
      .grid tr:nth-child(even) td { background:#1b1f29; }
      .grid.wide td, .grid.wide th { white-space:nowrap; }
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#1d2330; color:#a0aec0; font-size:12px; }
      footer { color:var(--sub); margin-top:24px; }
      a { color:#7ee; text-decoration:none; }
      a:hover { text-decoration:underline; }
    </style>
    """

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
{css}
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class="meta">Generated {ts}</div>

  <section>
    <h2>Squad — All Positions</h2>
    {squad_html}
  </section>

  {"".join(sections)}

  <footer>Saved by PosCalc • {ts}</footer>
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Score players (and output HTML report). Drag & drop files onto Run-PosCalc.cmd or run directly.")
    ap.add_argument("fm_text_file", type=Path, nargs="*", help="FM export(s) (.rtf or .txt). Leave empty to choose via file picker.")
    args = ap.parse_args()

    files = list(args.fm_text_file)
    if not files:
        # No args? Pop a file picker (Windows-friendly).
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            paths = filedialog.askopenfilenames(
                title="Select FM exports (.rtf or .txt)",
                filetypes=[("FM Exports", "*.rtf *.txt"), ("All files", "*.*")]
            )
            files = [Path(p) for p in paths]
        except Exception:
            pass

    if not files:
        print("No files provided.")
        return

    for fm_path in files:
        df = read_pipe_table(fm_path)

    # Ensure Age exists and is numeric (blank if missing in the export)
        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        else:
            df["Age"] = pd.NA

    # Build the wide table INCLUDING Age
        out_cols = {
            "Name": df["Name"],
            "Position": df["Position"],
            "Age": df["Age"],          # <-- keep Age in scores
        }
        for pos in PROFILES:
            out_cols[pos] = score_position(df, pos)

        scores = pd.DataFrame(out_cols)

    # --- scaling: map raw 0–100 to FIFA-like scale ---
        EXP_ALL = _exp_for_anchor(82)
        for pos in PROFILES.keys():
            scores[pos] = _piecewise_rescale_col(scores[pos])

    # Final rounding
        for pos in PROFILES.keys():
            scores[pos] = scores[pos].round().astype("Int64")

        best11 = build_best_xi(scores)
        subs9  = pick_best_subs(scores, best11, n=9)


    # --- output paths ---
        out_dir = fm_path.parent / "reports"
        out_dir.mkdir(exist_ok=True)
        csv_path  = out_dir / f"{fm_path.stem}_scores.csv"
        html_path = out_dir / f"{fm_path.stem}_report.html"

        scores.to_csv(csv_path, index=False)
        generate_html_report(scores, html_path, title=f"{fm_path.stem} — Squad Report",
                     best11=best11, subs=subs9)

        print(f"\nSaved CSV : {csv_path}")
        print(f"Saved HTML: {html_path}")
    

if __name__ == "__main__":
    main()

