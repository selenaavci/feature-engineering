import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feature Engineering Agent", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
.main-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; }
.subtitle { color: #6b7280; margin-top: 0; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Feature Engineering Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Ham veriyi tek tıkla modele hazır, anlamlı özelliklere dönüştür.</p>',
            unsafe_allow_html=True)

if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_transformed" not in st.session_state:
    st.session_state.df_transformed = None
if "log" not in st.session_state:
    st.session_state.log = []


def detect_column_types(df: pd.DataFrame) -> dict:
    types = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
            continue
        if s.dtype == object:
            sample = s.dropna().astype(str).head(20)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= max(3, int(0.7 * len(sample))):
                types[col] = "datetime"
                continue
            avg_len = sample.str.len().mean() if len(sample) else 0
            nunique = s.nunique(dropna=True)
            if avg_len > 20 or nunique > max(30, 0.5 * len(s)):
                types[col] = "text"
            else:
                types[col] = "categorical"
        else:
            types[col] = "categorical"
    return types


def log_action(msg: str):
    st.session_state.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def add_date_features(df, col, opts):
    s = pd.to_datetime(df[col], errors="coerce")
    created = []
    if "year" in opts:
        df[f"{col}_year"] = s.dt.year; created.append(f"{col}_year")
    if "month" in opts:
        df[f"{col}_month"] = s.dt.month; created.append(f"{col}_month")
    if "day" in opts:
        df[f"{col}_day"] = s.dt.day; created.append(f"{col}_day")
    if "dow" in opts:
        df[f"{col}_dayofweek"] = s.dt.dayofweek; created.append(f"{col}_dayofweek")
    if "weekend" in opts:
        df[f"{col}_is_weekend"] = s.dt.dayofweek.isin([5, 6]).astype("Int64"); created.append(f"{col}_is_weekend")
    if "quarter" in opts:
        df[f"{col}_quarter"] = s.dt.quarter; created.append(f"{col}_quarter")
    if "week" in opts:
        df[f"{col}_week"] = s.dt.isocalendar().week.astype("Int64"); created.append(f"{col}_week")
    return created


def add_text_features(df, col, opts):
    s = df[col].astype(str).fillna("")
    created = []
    if "length" in opts:
        df[f"{col}_length"] = s.str.len(); created.append(f"{col}_length")
    if "words" in opts:
        df[f"{col}_word_count"] = s.str.split().str.len(); created.append(f"{col}_word_count")
    if "empty" in opts:
        df[f"{col}_is_empty"] = (df[col].isna() | (s.str.strip() == "")).astype(int); created.append(f"{col}_is_empty")
    if "special" in opts:
        df[f"{col}_has_special"] = s.apply(lambda x: int(bool(re.search(r"[^\w\s]", x)))); created.append(f"{col}_has_special")
    if "upper" in opts:
        df[f"{col}_upper_ratio"] = s.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)); created.append(f"{col}_upper_ratio")
    return created


def add_categorical_features(df, col, opts):
    created = []
    if "frequency" in opts:
        freq = df[col].map(df[col].value_counts(normalize=True))
        df[f"{col}_freq"] = freq; created.append(f"{col}_freq")
    if "rare" in opts:
        counts = df[col].value_counts()
        rare = counts[counts < max(2, int(0.02 * len(df)))].index
        df[f"{col}_is_rare"] = df[col].isin(rare).astype(int); created.append(f"{col}_is_rare")
    if "group_size" in opts:
        df[f"{col}_group_size"] = df[col].map(df[col].value_counts()); created.append(f"{col}_group_size")
    if "onehot" in opts:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False).astype(int)
        for c in dummies.columns:
            df[c] = dummies[c]; created.append(c)
    return created


def add_numeric_features(df, col, opts, threshold=None, bins=4):
    s = df[col]
    created = []
    if "log" in opts:
        df[f"{col}_log"] = np.log1p(s.clip(lower=0)); created.append(f"{col}_log")
    if "missing_flag" in opts:
        df[f"{col}_is_missing"] = s.isna().astype(int); created.append(f"{col}_is_missing")
    if "bin" in opts:
        try:
            df[f"{col}_bin"] = pd.qcut(s, q=bins, labels=False, duplicates="drop")
            created.append(f"{col}_bin")
        except Exception:
            pass
    if "zscore" in opts:
        mu, sd = s.mean(), s.std(ddof=0)
        if sd and sd != 0:
            df[f"{col}_zscore"] = (s - mu) / sd
            created.append(f"{col}_zscore")
    if "threshold" in opts and threshold is not None:
        df[f"{col}_above_{threshold}"] = (s > threshold).astype("Int64")
        created.append(f"{col}_above_{threshold}")
    return created


with st.sidebar:
    st.header("1. Veri Yükle")
    uploaded = st.file_uploader("CSV veya XLSX", type=["csv", "xlsx"])
    use_sample = st.button("Örnek veri kullan", use_container_width=True)

    if uploaded is not None:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.session_state.df_original = df.copy()
        st.session_state.df_transformed = df.copy()
        st.session_state.log = []
        log_action(f"Veri yüklendi: {df.shape[0]} satır x {df.shape[1]} sütun")

    if use_sample:
        try:
            df = pd.read_csv("sample_data.csv")
            st.session_state.df_original = df.copy()
            st.session_state.df_transformed = df.copy()
            st.session_state.log = []
            log_action(f"Örnek veri yüklendi: {df.shape[0]} satır x {df.shape[1]} sütun")
        except FileNotFoundError:
            st.error("sample_data.csv bulunamadı.")

    st.divider()
    if st.session_state.df_transformed is not None:
        if st.button("Sıfırla", use_container_width=True):
            st.session_state.df_transformed = st.session_state.df_original.copy()
            st.session_state.log = []

if st.session_state.df_transformed is None:
    st.info("Soldan bir dataset yükle veya örnek veriyi kullan.")
    st.stop()

df = st.session_state.df_transformed
col_types = detect_column_types(st.session_state.df_original)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Genel Bakış", "Akıllı Öneriler", "Manuel Özellik",
    "Önizleme & Log", "Dışa Aktar",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Satır", df.shape[0])
    c2.metric("Sütun (Güncel)", df.shape[1])
    c3.metric("Orijinal Sütun", st.session_state.df_original.shape[1])
    c4.metric("Eksik Hücre", int(df.isna().sum().sum()))

    st.subheader("Sütun Tipleri")
    tdf = pd.DataFrame({
        "column": list(col_types.keys()),
        "detected_type": list(col_types.values()),
        "missing": [st.session_state.df_original[c].isna().sum() for c in col_types],
        "unique": [st.session_state.df_original[c].nunique() for c in col_types],
    })
    st.dataframe(tdf, use_container_width=True, hide_index=True)

    st.subheader("Veri Önizleme")
    st.dataframe(df.head(15), use_container_width=True)

with tab2:
    st.caption("Sistem, sütun tiplerine göre otomatik öneriler sunar. Uygulamak istediklerini seç ve tıkla.")

    date_cols = [c for c, t in col_types.items() if t == "datetime"]
    if date_cols:
        st.markdown("### Tarih Özellikleri")
        for col in date_cols:
            with st.expander(f"`{col}` için öneriler"):
                opts = st.multiselect(
                    "Seç:",
                    options=[("year", "Yıl"), ("month", "Ay"), ("day", "Gün"),
                             ("dow", "Haftanın günü"), ("weekend", "Hafta sonu flag"),
                             ("quarter", "Çeyrek"), ("week", "Yılın haftası")],
                    format_func=lambda x: x[1], key=f"date_{col}",
                )
                if st.button("Uygula", key=f"btn_date_{col}"):
                    keys = [o[0] for o in opts]
                    created = add_date_features(df, col, keys)
                    st.session_state.df_transformed = df
                    log_action(f"[date] {col} -> {', '.join(created)}")
                    st.success(f"Oluşturuldu: {', '.join(created)}")

    text_cols = [c for c, t in col_types.items() if t == "text"]
    if text_cols:
        st.markdown("### Metin Özellikleri")
        for col in text_cols:
            with st.expander(f"`{col}` için öneriler"):
                opts = st.multiselect(
                    "Seç:",
                    options=[("length", "Karakter uzunluğu"), ("words", "Kelime sayısı"),
                             ("empty", "Boş metin flag"), ("special", "Özel karakter var mı"),
                             ("upper", "Büyük harf oranı")],
                    format_func=lambda x: x[1], key=f"text_{col}",
                )
                if st.button("Uygula", key=f"btn_text_{col}"):
                    keys = [o[0] for o in opts]
                    created = add_text_features(df, col, keys)
                    st.session_state.df_transformed = df
                    log_action(f"[text] {col} -> {', '.join(created)}")
                    st.success(f"Oluşturuldu: {', '.join(created)}")

    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    if cat_cols:
        st.markdown("### Kategorik Özellikler")
        for col in cat_cols:
            with st.expander(f"`{col}` için öneriler"):
                opts = st.multiselect(
                    "Seç:",
                    options=[("frequency", "Frekans kodlama"), ("rare", "Nadir kategori flag"),
                             ("group_size", "Grup boyutu"), ("onehot", "One-hot encoding")],
                    format_func=lambda x: x[1], key=f"cat_{col}",
                )
                if st.button("Uygula", key=f"btn_cat_{col}"):
                    keys = [o[0] for o in opts]
                    created = add_categorical_features(df, col, keys)
                    st.session_state.df_transformed = df
                    log_action(f"[cat] {col} -> {', '.join(created)}")
                    st.success(f"Oluşturuldu: {', '.join(created)}")

    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    if num_cols:
        st.markdown("### Sayısal Özellikler")
        for col in num_cols:
            with st.expander(f"`{col}` için öneriler"):
                opts = st.multiselect(
                    "Seç:",
                    options=[("log", "Log dönüşüm"), ("missing_flag", "Eksik flag"),
                             ("bin", "Binning (çeyrekler)"), ("zscore", "Z-skor"),
                             ("threshold", "Eşik üstü flag")],
                    format_func=lambda x: x[1], key=f"num_{col}",
                )
                thr = None
                bins = 4
                if any(o[0] == "threshold" for o in opts):
                    thr = st.number_input(f"{col} için eşik", value=float(df[col].median()),
                                          key=f"thr_{col}")
                if any(o[0] == "bin" for o in opts):
                    bins = st.slider("Bin sayısı", 2, 10, 4, key=f"bins_{col}")
                if st.button("Uygula", key=f"btn_num_{col}"):
                    keys = [o[0] for o in opts]
                    created = add_numeric_features(df, col, keys, threshold=thr, bins=bins)
                    st.session_state.df_transformed = df
                    log_action(f"[num] {col} -> {', '.join(created)}")
                    st.success(f"Oluşturuldu: {', '.join(created)}")

with tab3:
    st.caption("Kodlama gerekmez. Sütunları seç, operasyonu seç.")
    num_cols_now = df.select_dtypes(include=[np.number]).columns.tolist()
    op = st.selectbox("Operasyon", ["Oran (a/b)", "Fark (a-b)", "Çarpım (a*b)",
                                    "Toplam (a+b)", "Flag (koşul)", "Metin içerir"])

    if op in ["Oran (a/b)", "Fark (a-b)", "Çarpım (a*b)", "Toplam (a+b)"]:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Sütun A", num_cols_now, key="ma")
        b = c2.selectbox("Sütun B", num_cols_now, key="mb")
        name = st.text_input("Yeni sütun adı", value=f"{a}_{op.split()[0].lower()}_{b}")
        if st.button("Oluştur", key="btn_manual_arith"):
            if op.startswith("Oran"):
                df[name] = df[a] / df[b].replace(0, np.nan)
            elif op.startswith("Fark"):
                df[name] = df[a] - df[b]
            elif op.startswith("Çarpım"):
                df[name] = df[a] * df[b]
            else:
                df[name] = df[a] + df[b]
            st.session_state.df_transformed = df
            log_action(f"[manual] {name} = {a} {op} {b}")
            st.success(f"`{name}` oluşturuldu.")

    elif op == "Flag (koşul)":
        col = st.selectbox("Sütun", num_cols_now, key="fcol")
        cond = st.selectbox("Koşul", [">", ">=", "<", "<=", "=="])
        val = st.number_input("Değer", value=float(df[col].median()))
        name = st.text_input("Yeni sütun adı", value=f"{col}_flag")
        if st.button("Oluştur", key="btn_manual_flag"):
            expr = {">": df[col] > val, ">=": df[col] >= val, "<": df[col] < val,
                    "<=": df[col] <= val, "==": df[col] == val}[cond]
            df[name] = expr.astype(int)
            st.session_state.df_transformed = df
            log_action(f"[manual] {name} = ({col} {cond} {val})")
            st.success(f"`{name}` oluşturuldu.")

    elif op == "Metin içerir":
        txt_cols = df.select_dtypes(include="object").columns.tolist()
        col = st.selectbox("Sütun", txt_cols, key="tcol")
        kw = st.text_input("Aranacak kelime", value="good")
        name = st.text_input("Yeni sütun adı", value=f"{col}_contains_{kw}")
        if st.button("Oluştur", key="btn_manual_text"):
            df[name] = df[col].astype(str).str.contains(kw, case=False, na=False).astype(int)
            st.session_state.df_transformed = df
            log_action(f"[manual] {name} = {col}.contains('{kw}')")
            st.success(f"`{name}` oluşturuldu.")

with tab4:
    original_cols = set(st.session_state.df_original.columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    st.subheader(f"Yeni Özellikler ({len(new_cols)})")
    if new_cols:
        st.dataframe(df[new_cols].head(20), use_container_width=True)
    else:
        st.info("Henüz yeni özellik yok.")

    st.subheader("Dönüşüm Günlüğü")
    if st.session_state.log:
        for line in st.session_state.log:
            st.text(line)
    else:
        st.caption("Henüz işlem yok.")

with tab5:
    st.caption("Dönüştürülmüş dataseti indir.")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("CSV indir", csv, "transformed.csv", "text/csv")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="data")
        pd.DataFrame({"log": st.session_state.log}).to_excel(w, index=False, sheet_name="log")
    st.download_button("Excel indir", buf.getvalue(), "transformed.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
