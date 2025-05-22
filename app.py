# ╔══════════════════════════════════════════════════════════════╗
# ║  YVS Dealers – Sales, GST & Rewards Explorer                ║
# ║  • Merges uploaded files (skipping first row) into merged.xlsx ║
# ║  • Uses merged.xlsx by default                                   ║
# ║  • Filters: Date, Region, Category, Party, Registration         ║
# ║  • KPI Cards, Charts, and Reward Eligibility with multiple gifts║
# ╚══════════════════════════════════════════════════════════════╝
import streamlit as st
import pandas as pd
import plotly.express as px
import pathlib, re, warnings

warnings.filterwarnings("ignore")

# 1. PAGE CONFIGURATION
st.set_page_config("YVS Sales Dashboard", "📊", layout="wide")
st.title("📊  YVS Dealers : Sales, GST & Rewards")

# 2. SESSION STATE FOR DEDUPLICATION
if "seen_files" not in st.session_state:
    st.session_state.seen_files = set()

warnings.filterwarnings("ignore")

# ─── project dirs ───────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.resolve()
MERGED_XLSX = ROOT / "merged.xlsx"

# 3. FILE UPLOAD & MERGE
uploads = st.file_uploader(
    "📂 Upload CSV/Excel files (skip headers, merge)",
    type=["csv","txt","xlsx","xls"],
    accept_multiple_files=True
)

if uploads:
    frames = []
    for fp in uploads:
        key = (fp.name, fp.size)
        if key not in st.session_state.seen_files:
            st.session_state.seen_files.add(key)
            # skip first row of raw
            suf = pathlib.Path(fp.name).suffix.lower()
            if suf in {".csv",".txt"}:
                df_raw = pd.read_csv(fp, skiprows=1, encoding="ISO-8859-1")
            else:
                xls = pd.ExcelFile(fp)
                df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0], skiprows=1)
            frames.append(df_raw)
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.to_excel(MERGED_XLSX, index=False)
        st.success(f"💾 merged.xlsx saved ({len(df):,} rows)")
    else:
        st.info("No new files—using existing merged.xlsx")
        if MERGED_XLSX.exists():
            df = pd.read_excel(MERGED_XLSX)
        else:
            st.error("❌ No merged.xlsx found—upload files first.")
            st.stop()
else:
    if MERGED_XLSX.exists():
        df = pd.read_excel(MERGED_XLSX)
        st.info("Using existing merged.xlsx")
    else:
        st.error("❌ No data: upload files to create merged.xlsx")
        st.stop()

# 4. CLEAN HEADERS & TYPES
def clean(h): return re.sub(r"\s+"," ",h.strip()).title()
df.columns = [clean(c) for c in df.columns]
# required columns
req = ["Bill Date","Bill Amount","Party Name","Location","Cash / Credit"]
miss = [c for c in req if c not in df.columns]
if miss:
    st.error("Missing columns: " + ", ".join(miss))
    st.stop()
# types
df["Bill Date"] = pd.to_datetime(df["Bill Date"], errors="coerce")
df["Bill Amount"] = pd.to_numeric(df["Bill Amount"], errors="coerce")
# tax
tax_cols = {"Cgst","Sgst","Igst"}
if tax_cols.issubset(df.columns) and "Tax" not in df.columns:
    df["Tax"] = df[list(tax_cols)].sum(axis=1)
# registration status
if "Gstin" in df.columns:
    df["RegStatus"] = df["Gstin"].astype(str).str.strip().str.len().ge(5)
    df["RegStatus"] = df["RegStatus"].map({True:"Registered",False:"Un-Registered"})
else:
    df["RegStatus"] = "Un-Registered"

# 5. DATE FILTER
c1,c2 = st.columns(2)
with c1:
    start = st.date_input("Start date", df["Bill Date"].min())
with c2:
    end = st.date_input("End date", df["Bill Date"].max())
filtered = df.query("@start <= `Bill Date` <= @end").copy()

# 6. SIDEBAR FILTERS
st.sidebar.header("🔍 Filters")
regn = st.sidebar.multiselect("Region", sorted(filtered["Location"].dropna().astype(str).unique()))
cat_choice = st.sidebar.radio("Category", ["All","Cash","Credit"], horizontal=True)
party = st.sidebar.multiselect("Party Name", sorted(filtered["Party Name"].dropna().astype(str).unique()))
reg_status = st.sidebar.radio("Registration", ["All","Registered","Un-Registered"], horizontal=True)
mask = []
if regn:      mask.append("Location in @regn")
if cat_choice != "All": mask.append("`Cash / Credit`.str.lower()==@cat_choice.lower()")
if party:     mask.append("`Party Name` in @party")
if reg_status != "All": mask.append("RegStatus==@reg_status")
cur = filtered.query(" and ".join(mask)) if mask else filtered

# 7. KPI CARDS
k1,k2,k3 = st.columns(3)
k1.metric("Invoices", f"{len(cur):,}")
k2.metric("Net Sales", f"₹{cur['Bill Amount'].sum():,.0f}")
k3.metric("GST", f"₹{cur['Tax'].sum():,.0f}" if "Tax" in cur else "—")

# 8. REWARD ELIGIBILITY
st.subheader("🎁 Reward Eligibility – Dhamaka")
tiers = pd.DataFrame({
    'Min':[0,25001,50001,75000],
    'Max':[25000,50000,75000,100000],
    'Gift':['REWARD 1','REWARD 2','REWARD 3','REWARD 4']
})
party_tot = cur.groupby('Party Name', as_index=False)['Bill Amount'].sum()
mobiles = cur.groupby('Party Name')['Mobile'].first().reset_index()
party_tot = party_tot.merge(mobiles, on='Party Name', how='left')
def assign(x):
    df2=tiers[tiers.Min<=x]
    tier=df2.loc[df2.Min.idxmax()]
    cnt=int(x//tier.Max) if x>=tier.Max else 1
    return pd.Series({'Reward Type':tier.Gift,'Quantity':cnt})
ginfo=party_tot['Bill Amount'].apply(assign)
reward=party_tot.join(ginfo)[ginfo['Quantity']>0]
# add S.No.
reward.insert(0,'S.No.', range(1,len(reward)+1))
# show filters
gtypes=st.multiselect("Filter Reward Types", sorted(reward['Reward Type'].unique()))
if gtypes: reward=reward[reward['Reward Type'].isin(gtypes)]
min_sal=st.slider("Min Total Sales", int(reward['Bill Amount'].min()), int(reward['Bill Amount'].max()), int(reward['Bill Amount'].min()))
reward=reward[reward['Bill Amount']>=min_sal]
st.dataframe(reward[['S.No.','Party Name','Bill Amount','Reward Type','Quantity','Mobile']])
st.download_button("Download Rewards CSV", reward.to_csv(index=False).encode(),"rewards.csv","text/csv")

# 9. VISUALS
if "Cash / Credit" in cur:
    cat_df=cur.groupby("Cash / Credit",as_index=False)["Bill Amount"].sum()
    st.subheader("Sales by Category")
    st.plotly_chart(px.bar(cat_df,x="Cash / Credit",y="Bill Amount",text_auto=".2s"),use_container_width=True)
if "Location" in cur:
    reg_df=cur.groupby("Location",as_index=False)["Bill Amount"].sum()
    st.subheader("Sales by Region")
    st.plotly_chart(px.pie(reg_df,names="Location",values="Bill Amount",hole=.45),use_container_width=True)
cur["Year-Month"]=cur["Bill Date"].dt.to_period("M")
trend=cur.groupby(cur["Year-Month"].dt.strftime("%Y-%b"))["Bill Amount"].sum().reset_index()
st.subheader("Monthly Sales Trend")
st.plotly_chart(px.line(trend,x="Year-Month",y="Bill Amount",markers=True),use_container_width=True)
