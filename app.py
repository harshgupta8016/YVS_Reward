import streamlit as st
import pandas as pd
import duckdb
import pathlib
import warnings
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
import time
import io
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm

# ─────────────────────────── Page & global settings ───────────────────────────
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).parent.resolve()
PARQUET = ROOT / "merged.parquet"
DATE_COL = "Bill Date"

STR_COLS = {"GSTIN", "Mobile", "Address Line1", "Address Line2",
            "Address Line3", "City", "PIN"}

slabs = [
    (0, 24_999,  "No Reward"),
    (25_000, 49_999, "REWARD 1"),
    (50_000, 74_999, "REWARD 2"),
    (75_000, 99_999, "REWARD 3"),
    (100_000, float("inf"), "REWARD 4"),
]

def assign_reward(total: float):
    for lo, hi, tier in slabs:
        if lo == 0 and lo <= total <= hi:
            return tier, 0
        if lo > 0 and lo <= total <= hi:
            return tier, int(total // lo)
    return None, 0

# ─────── PDF helper – portrait, cut-ready boxes WITH 2 mm gaps ────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

def df_to_pdf_bytes(df: pd.DataFrame) -> bytes:
    """Portrait A4, 3×8 labels, 2 mm gaps between boxes, 10 pt font."""
    if df.empty:
        return b""

    # ------------ layout constants -------------
    PAGE_W, PAGE_H     = A4          # 210 × 297 mm
    MARGIN             = 8 * mm
    GAP                = 2 * mm      # gap between boxes
    usable_w           = PAGE_W - 2*MARGIN
    usable_h           = PAGE_H - 2*MARGIN

    col_w_box          = (usable_w - 2*GAP) / 3     # 3 boxes + 2 gaps
    row_h_box          = (usable_h - 7*GAP) / 8     # 8 boxes + 7 gaps

    # ------------ style -------------
    base   = getSampleStyleSheet()
    pstyle = ParagraphStyle("lbl",
                            parent   = base["Normal"],
                            fontSize = 10,
                            leading  = 11)

    # ------------ build label text -------------
    labels = []
    for _, r in df.iterrows():
        name  = str(r["Party Name"]).title()

        phone = str(r.get("Phone", "") or "").strip()
        phone = phone if (phone.isdigit() and phone != "0") else ""

        pin   = str(r.get("PIN", "") or "").strip()
        pin   = pin if (pin.isdigit() and pin != "000000") else ""

        addr  = str(r.get("Address", "") or "").title()
        city  = str(r.get("City", "") or "").title()
        addr_full = ", ".join(filter(None, [addr, city]))
        if pin:
            addr_full += f" – {pin}"

        parts = [
            f"<b>Name:</b> {name}",
            f"<b>Address:</b> {addr_full}",
            f"<b>Phone:</b> {phone}",
        ]
        labels.append("<br/>".join(parts))

    while len(labels) % 24:
        labels.append("")

    # ------------ create PDF -------------
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN)

    story = []
    paras = [Paragraph(t, pstyle) for t in labels]

    # Utility to mark label-cell positions (even rows/cols)
    def label_pos(row, col):
        return (row*2, col*2)   # convert logical 0‒7/0‒2 → physical row/col

    for idx in range(0, len(paras), 24):
        chunk = paras[idx:idx+24]

        # Build 15×5 data grid (label / gap columns & rows)
        data = [["" for _ in range(5)] for _ in range(15)]
        k = 0
        for r in range(8):
            for c in range(3):
                pr = label_pos(r, c)
                data[pr[0]][pr[1]] = chunk[k]; k += 1

        col_widths = [col_w_box, GAP, col_w_box, GAP, col_w_box]
        row_heights = []
        for r in range(15):
            row_heights.append(row_h_box if r % 2 == 0 else GAP)

        tbl = Table(data, colWidths=col_widths, rowHeights=row_heights)

        # style: border around each label cell only
        styles = [("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                  ("ALIGN",  (0,0), (-1,-1), "LEFT")]
        for r in range(0,15,2):          # even rows
            for c in range(0,5,2):       # even cols
                styles.append(("BOX", (c,r), (c,r), 0.8, colors.black))

        tbl.setStyle(TableStyle(styles))
        story.append(tbl)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

# ──────────────────────────────── Caching helpers ─────────────────────────────
@st.cache_data(ttl=3600)
def build_parquet_from_uploads(files):
    dfs = []
    for f in files:
        ext = pathlib.Path(f.name).suffix.lower()
        tmp = (pd.read_excel(f, engine="openpyxl", skiprows=1)
               if ext in {".xlsx", ".xls"}
               else pd.read_csv(f, skiprows=1, encoding="ISO-8859-1"))
        for c in STR_COLS & set(tmp.columns):
            tmp[c] = tmp[c].astype("string")
        if DATE_COL in tmp.columns:
            tmp[DATE_COL] = pd.to_datetime(tmp[DATE_COL], dayfirst=True,
                                           errors="coerce").dt.date
        dfs.append(tmp)
    if not dfs:
        return 0, 0
    df = pd.concat(dfs, ignore_index=True)
    for c in STR_COLS:
        df[c] = df.get(c, pd.NA).astype("string")
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True,
                                      errors="coerce").dt.date
    df.to_parquet(PARQUET, index=False)
    return len(dfs), len(df)

@st.cache_data(ttl=3600)
def query_duckdb(sql: str, params=None):
    con = duckdb.connect()
    return con.execute(sql, params or []).df()

# ─────────────────────────────── Custom CSS ────────────────────────────────
st.markdown("""
<style>
.ag-root-wrapper,.ag-body-viewport{overflow-x:auto!important}
[data-testid="metric-container"]{margin:0 0.5rem!important;padding:0.5rem!important}
[data-testid="metric-container"] .stMetricLabel,[data-testid="metric-container"] .stMetricValue{padding:0!important;white-space:nowrap}
</style>""", unsafe_allow_html=True)

# ───────────────────────────── Header & upload ─────────────────────────────
title_col, upload_col = st.columns([4,1])
with title_col:
    st.markdown("<h1 style='margin:0;'>📊 YVS Dealers – Analytics & Rewards</h1>",
                unsafe_allow_html=True)
with upload_col:
    uploaded = st.file_uploader("Upload Excel/CSV (skip header row)",
                                type=["csv","xls","xlsx"],
                                accept_multiple_files=True,
                                label_visibility="collapsed")
    if uploaded:
        try:
            nf, nr = build_parquet_from_uploads(uploaded)
            st.success(f"Merged {nf} file(s), {nr} rows.")
        except Exception as e:
            st.error(f"Upload error: {e}")
    elif not PARQUET.exists():
        st.error("No data available. Please upload files.")
        st.stop()
    else:
        ph = st.empty()
        ph.info("Using cached data from merged.parquet")
        time.sleep(3); ph.empty()

# ─────────────────────────────── Sidebar filters ─────────────────────────────
with st.sidebar:
    st.markdown("<style>.sidebar .stMarkdown{padding:0.5rem 1rem}"
                ".sidebar .stExpander>div{background:#f0f2f6;border-radius:8px;"
                "margin-bottom:1rem}</style>", unsafe_allow_html=True)
    st.markdown("## 🎛️ Filters", unsafe_allow_html=True)

    cols = pd.read_parquet(PARQUET).columns
    if DATE_COL in cols:
        with st.expander("📅 Date Range", True):
            bnd = query_duckdb(f'SELECT MIN("{DATE_COL}") mn, MAX("{DATE_COL}") mx '
                               'FROM PARQUET_scan($1)', [str(PARQUET)])
            start = st.date_input("Start", bnd.at[0,'mn'])
            end   = st.date_input("End",   bnd.at[0,'mx'])
    else:
        start = end = None

    def get_distinct(c):
        sql, ps = f'SELECT DISTINCT "{c}" FROM PARQUET_scan($1)', [str(PARQUET)]
        if start and end:
            sql += f' WHERE "{DATE_COL}" BETWEEN ? AND ?'; ps += [start, end]
        return sorted(query_duckdb(sql, ps)[c].dropna().astype(str).unique())

    with st.expander("🏬 Shop, Party & City"):
        shop  = st.multiselect("Shop",  get_distinct("Shop"))
        party = st.multiselect("Party", get_distinct("Party Name"))
        city  = st.multiselect("City",  get_distinct("City"))

    with st.expander("🔖 Registration & Category"):
        reg = st.radio("Registration", ["All","Registered","Un-registered"])
        cat = st.radio("Category", ["All","CASH","CREDIT"])

    with st.expander("💰 Sales & Rewards"):
        max_sales_val = int(pd.read_parquet(PARQUET)["Bill Amount"].sum())
        min_sales = st.number_input("Min Sales (₹)", 0, max_sales_val, 0, step=10000)
        max_sales = st.number_input("Max Sales (₹)", 0, max_sales_val,
                                    max_sales_val, step=10000)
        tier_sel = st.multiselect("Reward Tier",
                                  [t for _,_,t in slabs] + ["No Reward"])

# ─────────────────────────── Main query / DataFrame ──────────────────────────
params, conds = [str(PARQUET)], []
if start and end:
    conds.append(f'"{DATE_COL}" BETWEEN ? AND ?'); params += [start, end]
if shop:
    conds.append(f'"Shop" IN ({",".join(["?"]*len(shop))})'); params += shop
if party:
    conds.append(f'"Party Name" IN ({",".join(["?"]*len(party))})'); params += party
if reg != "All":
    conds.append('(LENGTH(CAST(GSTIN AS VARCHAR))>4)=?')
    params.append(1 if reg=="Registered" else 0)
if cat != "All":
    conds.append('"Cash / Credit"=?'); params.append(cat)

where = " WHERE "+ " AND ".join(conds) if conds else ""
query = f'''
SELECT "Party Name",
       SUM("Bill Amount")   AS total_sales,
       MAX("Mobile")        AS Mobile,
       MAX("Address Line1") AS AL1,
       MAX("Address Line2") AS AL2,
       MAX("Address Line3") AS AL3,
       MAX("City")          AS City,
       MAX("PIN")           AS PIN
FROM PARQUET_scan($1)
{where}
GROUP BY "Party Name"'''

party_df = query_duckdb(query, params)
party_df["Phone"]   = party_df["Mobile"].where(party_df["Mobile"].str.isdigit(), None)
party_df["Address"] = party_df[["AL1","AL2","AL3"]].fillna("").agg(", ".join, axis=1)
party_df["numeric_sales"] = party_df["total_sales"].astype(float)
party_df[["Reward","Count"]] = party_df["numeric_sales"].apply(lambda x: pd.Series(assign_reward(x)))

# ───────────────────────────── KPIs ─────────────────────────────
st.markdown("---")
kpis = [("Net Sales", f"₹{party_df['numeric_sales'].sum():,.2f}")]
kpis += [(tier, int(party_df.loc[party_df["Reward"]==tier,"Count"].sum()))
         for _,_,tier in slabs if tier!="No Reward"]
for c,(lbl,val) in zip(st.columns(len(kpis)), kpis):
    c.metric(lbl, val)

# ─────────────────────── Prepare display DataFrame ───────────────────────
flt = party_df[party_df["numeric_sales"].between(min_sales, max_sales)]
if tier_sel:
    m = pd.Series(False, index=flt.index)
    for t in tier_sel:
        m |= flt["Reward"].eq(t) if t!="No Reward" else flt["Reward"].isna()
    flt = flt[m]

disp = flt[["Party Name","numeric_sales","Reward","Count",
            "Phone","Address","City","PIN"]].sort_values("Party Name").reset_index(drop=True)
disp.insert(0,"S.No.", range(1,len(disp)+1))
disp["PIN"] = disp["PIN"].apply(lambda x: f"{int(float(x)):06d}"
                                if pd.notna(x) and str(x).replace(".0","").isdigit() else None)
disp = disp.rename(columns={"numeric_sales":"Sales"})
disp["Sales"] = disp["Sales"].apply(lambda x: f"(₹ {x:,.2f})")

# ───────────────────────────── AgGrid ─────────────────────────────
gb = GridOptionsBuilder.from_dataframe(disp)
gb.configure_selection("multiple", use_checkbox=True, header_checkbox=True)
gb.configure_pagination(paginationPageSize=25)
grid_opts = gb.build()

st.markdown("---")
grid_res = AgGrid(disp, gridOptions=grid_opts,
                  data_return_mode=DataReturnMode.AS_INPUT,
                  update_mode=GridUpdateMode.MODEL_CHANGED,
                  height=600, width="100%", fit_columns_on_grid_load=False)
sel_df = pd.DataFrame(grid_res.get("selected_rows", []))\
           .drop(columns=["_selectedRowNodeInfo"], errors="ignore")

# ───────────────────────── Downloads (4 buttons) ─────────────────────────
pdf_all = df_to_pdf_bytes(disp)
pdf_sel = df_to_pdf_bytes(sel_df)

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.download_button("Download All (CSV)",
                       disp.to_csv(index=False).encode(),
                       "rewards.csv", "text/csv")
with c2:
    st.download_button("Download Selected (CSV)",
                       sel_df.to_csv(index=False).encode(),
                       "selected.csv", "text/csv",
                       disabled=sel_df.empty)
with c3:
    st.download_button("Download All (PDF)",
                       pdf_all, "rewards.pdf", "application/pdf")
with c4:
    st.download_button("Download Selected (PDF)",
                       pdf_sel, "selected.pdf", "application/pdf",
                       disabled=sel_df.empty)
