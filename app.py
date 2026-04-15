import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Feature Engineering Agent", page_icon="🛠️", layout="wide")

st.title("Feature Engineering Agent")
st.caption("Ham veriyi tek tıkla modele hazır, anlamlı özelliklere dönüştür. "
           "Her dönüşümün 'Nedir?' ve 'Ne zaman?' açıklaması ile Makine Öğrenmesi bilgisi gerekmez.")

if "df_original" not in st.session_state:
    st.session_state.df_original = None
if "df_transformed" not in st.session_state:
    st.session_state.df_transformed = None
if "log" not in st.session_state:
    st.session_state.log = []
if "type_overrides" not in st.session_state:
    st.session_state.type_overrides = {}
if "data_version" not in st.session_state:
    st.session_state.data_version = 0


METHOD_DOCS = {
    "year":    ("Yıl bilgisini çıkarır (örn. 2023).",
                "Verinin yıllara göre değişip değişmediğini incelemek istediğinde."),
    "month":   ("Ay bilgisini çıkarır (1–12).",
                "Mevsimsellik veya aylık trend aramak istediğinde (ör. satış, kampanya)."),
    "day":     ("Ayın günü (1–31).",
                "Ay başı / ay sonu gibi günlük desenler aradığında."),
    "dow":     ("Haftanın günü (0=Pzt, 6=Paz).",
                "Hafta içi/hafta sonu davranış farkını yakalamak istediğinde."),
    "weekend": ("Hafta sonu mu? Evet/Hayır (1/0) değeri üretir.",
                "Hafta sonları farklı davranış bekliyorsan hızlı bir sinyaldir."),
    "quarter": ("Yılın çeyreği (Q1–Q4).",
                "Çeyrek bazlı finans/planlama analizleri için uygundur."),
    "week":    ("Yılın haftası (1–53).",
                "Haftalık dönemsellik (ör. kampanya haftaları) önemliyse."),

    "length":  ("Metnin karakter sayısını hesaplar.",
                "Kısa/uzun metinlerin davranışı farklıysa (ör. kısa yorum = olumsuz)."),
    "words":   ("Kelime sayısını hesaplar.",
                "Detaylı açıklamaların performansa etkisini ölçmek istediğinde."),
    "empty":   ("Metin boş veya eksik mi? 1/0 bayrağı üretir.",
                "Yorum bırakmayan müşteriler farklı davranıyorsa önemli bir sinyaldir."),
    "special": ("Metinde özel karakter (!, @, #, …) var mı?",
                "Abartılı ifadeler / spam içeriği ayırmak istediğinde."),
    "upper":   ("Metindeki büyük harf oranı.",
                "BÜYÜK HARFLE yazan kullanıcıların (ör. öfkeli yorumlar) sinyali olabilir."),

    "frequency":  ("Her kategori için görülme oranı (0–1). "
                   "Örn. 'TR' müşterilerin %40'ıysa değer 0.40 olur.",
                   "Modele kategoriyi sayısal olarak tanıtmanın basit bir yolu. "
                   "Çok fazla kategori varsa one-hot yerine tercih edilir."),
    "rare":       ("Toplam verinin %2'sinden az görülen kategoriler için 1, diğerleri 0.",
                   "Az sayıda görülen değerlerin modeli yanıltmasını önler."),
    "group_size": ("Her kategorinin kaç satırda geçtiği.",
                   "Popüler/az kullanılan grupları ayırt etmek için."),
    "onehot":     ("Her kategori değeri için ayrı bir 0/1 sütun üretir.",
                   "Kategori sayısı az (≤10) ise en güvenli ve standart kodlamadır. "
                   "Çok fazla kategori varsa sütun patlamasına yol açar — frekans kullan."),

    "log":          ("log(1+x) dönüşümü uygular.",
                     "Gelir, fiyat gibi çok çarpık (uçları uzun) sayılar için. "
                     "Uç değerlerin etkisini azaltır, dağılımı normale yaklaştırır."),
    "missing_flag": ("Değer eksikse 1, değilse 0.",
                     "Eksik olmak başlı başına bilgiyse (örn. geliri bildirmeyenler). "
                     "Eksik doldurma yaptığında bu bayrak sinyali korur."),
    "bin":          ("Sayıyı eşit büyüklükte gruplara (çeyrek/beşli) böler.",
                     "Sürekli sayıyı 'düşük/orta/yüksek' gibi gruplara indirgemek için. "
                     "Doğrusal olmayan ilişkileri yakalar."),
    "zscore":       ("(x - ortalama) / standart sapma. Standart normalleştirme.",
                     "Farklı ölçekteki sayıları karşılaştırılabilir hale getirir. "
                     "Mesafeye dayalı modeller (KNN, kümeleme) için önemli."),
    "threshold":    ("Eşik üstü için 1, altı için 0 bayrağı üretir.",
                     "İş kuralları (ör. 'gelir > 50K ise premium') modellemek için."),
}


def show_method(key):
    nedir, nezaman = METHOD_DOCS[key]
    st.markdown(f"**Nedir?** {nedir}")
    st.markdown(f"**Ne zaman?** {nezaman}")


DATE_NAME_HINTS = ("tarih", "date", "datum", "zaman", "time", "day",
                   "_dt", "dt_", "created", "updated", "signup", "birth", "dogum")


DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
                "%d-%m-%Y", "%d.%m.%Y", "%Y.%m.%d", "%Y-%m-%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]


def _parse_date_ratio(series: pd.Series) -> float:
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(200)
    if len(sample) == 0:
        return 0.0
    import warnings
    best = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kwargs in ({}, {"dayfirst": True}, {"yearfirst": True}):
            try:
                parsed = pd.to_datetime(sample, errors="coerce", **kwargs)
                best = max(best, int(parsed.notna().sum()))
            except Exception:
                pass
        for fmt in DATE_FORMATS:
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format=fmt)
                best = max(best, int(parsed.notna().sum()))
            except Exception:
                pass
    return best / len(sample)


def detect_column_types(df: pd.DataFrame) -> dict:
    types = {}
    for col in df.columns:
        s = df[col]
        name_hint = any(h in str(col).lower() for h in DATE_NAME_HINTS)

        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
            continue

        if s.dtype == object:
            ratio = _parse_date_ratio(s)
            threshold = 0.2 if name_hint else 0.7
            if ratio >= threshold:
                types[col] = "datetime"
                continue

        if name_hint and s.dtype == object:
            types[col] = "datetime"
            continue

        if pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
            continue

        if s.dtype == object:
            sample = s.dropna().astype(str).head(50)
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


def read_any(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    if name.endswith(".xml"):
        return pd.read_xml(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    raise ValueError(f"Desteklenmeyen dosya tipi: {name}")


def add_date_features(df, col, opts):
    s = pd.to_datetime(df[col], errors="coerce")
    created = []
    mapping = {
        "year": (f"{col}_year", s.dt.year),
        "month": (f"{col}_month", s.dt.month),
        "day": (f"{col}_day", s.dt.day),
        "dow": (f"{col}_dayofweek", s.dt.dayofweek),
        "weekend": (f"{col}_is_weekend", s.dt.dayofweek.isin([5, 6]).astype("Int64")),
        "quarter": (f"{col}_quarter", s.dt.quarter),
        "week": (f"{col}_week", s.dt.isocalendar().week.astype("Int64")),
    }
    for k in opts:
        name, vals = mapping[k]
        df[name] = vals
        created.append(name)
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
        df[f"{col}_freq"] = df[col].map(df[col].value_counts(normalize=True)); created.append(f"{col}_freq")
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


def render_method_group(title, col, options_spec, apply_fn, extra_inputs=None):
    st.markdown(f"#### {title}")
    chosen = []
    for key, label in options_spec:
        cols = st.columns([1, 6])
        with cols[0]:
            checked = st.checkbox(" ", key=f"{col}_{key}_chk",
                                  label_visibility="collapsed")
        with cols[1]:
            with st.expander(label, expanded=False):
                show_method(key)
        if checked:
            chosen.append(key)
    extra = {}
    if extra_inputs:
        extra = extra_inputs(chosen)
    if st.button("Seçilenleri Uygula", key=f"apply_{col}_{title}",
                 type="primary", use_container_width=True):
        if not chosen:
            st.warning("En az bir yöntem seç.")
        else:
            created = apply_fn(chosen, extra)
            if created:
                st.session_state.df_transformed = st.session_state.df_transformed
                log_action(f"[{title}] {col} -> {', '.join(created)}")
                st.success(f"Oluşturuldu: {', '.join(created)}")
            else:
                st.warning("Sütun uygun değil, özellik üretilemedi.")


with st.sidebar:
    st.header("Veri Yükle")
    st.caption("Desteklenen formatlar: CSV, XLSX, XLS, XML, JSON. Maks. 5 GB.")
    uploaded = st.file_uploader("Dosya seç", type=["csv", "xlsx", "xls", "xml", "json"])
    use_sample = st.button("Örnek veri kullan", use_container_width=True)

    if uploaded is not None:
        upload_key = (uploaded.name, getattr(uploaded, "size", None))
        if st.session_state.get("_uploaded_key") != upload_key:
            try:
                df = read_any(uploaded)
                st.session_state.df_original = df.copy()
                st.session_state.df_transformed = df.copy()
                st.session_state.log = []
                st.session_state.type_overrides = {}
                st.session_state.data_version += 1
                st.session_state._uploaded_key = upload_key
                log_action(f"Veri yüklendi: {df.shape[0]} satır x {df.shape[1]} sütun")
            except Exception as e:
                st.error(f"Yükleme hatası: {e}")

    if use_sample:
        try:
            df = pd.read_csv("sample_data.csv")
            st.session_state.df_original = df.copy()
            st.session_state.df_transformed = df.copy()
            st.session_state.log = []
            st.session_state.type_overrides = {}
            st.session_state.data_version += 1
            log_action(f"Örnek veri yüklendi: {df.shape[0]} satır x {df.shape[1]} sütun")
        except FileNotFoundError:
            st.error("sample_data.csv bulunamadı.")

    st.divider()
    if st.session_state.df_transformed is not None:
        if st.button("Sıfırla", use_container_width=True):
            st.session_state.df_transformed = st.session_state.df_original.copy()
            st.session_state.log = []
            st.session_state.type_overrides = {}

    with st.expander("ℹ️ Nasıl kullanılır?"):
        st.markdown(
            "1. Dosyanı yükle veya örnek veriyi kullan.\n"
            "2. **Akıllı Öneriler** sekmesinde sütunlarını gör; her yöntemin "
            "'Nedir?' / 'Ne zaman?' açıklamasını oku.\n"
            "3. İstediklerini seç → *Seçilenleri Uygula*.\n"
            "4. **Önizleme** ile yeni sütunları kontrol et.\n"
            "5. **Dışa Aktar** ile dönüştürülmüş veriyi indir."
        )

if st.session_state.df_transformed is None:
    st.info("Soldan bir dataset yükle veya örnek veriyi kullan.")
    st.stop()

df = st.session_state.df_transformed
col_types = detect_column_types(st.session_state.df_original)
for _c, _t in st.session_state.type_overrides.items():
    if _c in col_types:
        col_types[_c] = _t

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
    st.caption("Tip yanlış algılandıysa aşağıdan düzeltebilirsin. Değişiklik diğer sekmeleri anında günceller.")
    TYPE_OPTS = ["numeric", "categorical", "datetime", "text"]
    edit_rows = []
    for c in col_types:
        edit_rows.append({
            "column": c,
            "type": col_types[c],
            "missing": int(st.session_state.df_original[c].isna().sum()),
            "unique": int(st.session_state.df_original[c].nunique()),
        })
    edited = st.data_editor(
        pd.DataFrame(edit_rows),
        use_container_width=True, hide_index=True, num_rows="fixed",
        disabled=["column", "missing", "unique"],
        column_config={"type": st.column_config.SelectboxColumn(
            "type", options=TYPE_OPTS, required=True,
        )},
        key=f"type_editor_v{st.session_state.data_version}",
    )
    new_overrides = {r["column"]: r["type"] for _, r in edited.iterrows()}
    if new_overrides != {c: col_types[c] for c in col_types}:
        st.session_state.type_overrides = {
            c: t for c, t in new_overrides.items() if c in col_types
        }
        st.rerun()

    with st.expander("Sütun tipleri ne anlama geliyor?"):
        st.markdown(
            "- **numeric**: sayısal değer (fiyat, yaş, miktar)\n"
            "- **categorical**: sınırlı sayıda kategori (ülke, segment)\n"
            "- **datetime**: tarih / saat\n"
            "- **text**: serbest metin (yorum, açıklama)\n\n"
            "Otomatik tespit; yanlış olduğunu düşünürsen manuel sekmeyi kullanabilirsin."
        )

    st.subheader("Veri Önizleme")
    st.dataframe(df.head(15), use_container_width=True)

with tab2:
    st.caption("Sütununu seç, uygun yöntemleri incele — her birinin açıklaması var.")

    date_cols = [c for c, t in col_types.items() if t == "datetime"]
    text_cols = [c for c, t in col_types.items() if t == "text"]
    cat_cols  = [c for c, t in col_types.items() if t == "categorical"]
    num_cols  = [c for c, t in col_types.items() if t == "numeric"]

    if date_cols:
        st.markdown("### Tarih Sütunları")
        for col in date_cols:
            with st.container(border=True):
                st.markdown(f"**Sütun:** `{col}`")
                spec = [("year","Yıl"),("month","Ay"),("day","Gün"),
                        ("dow","Haftanın günü"),("weekend","Hafta sonu bayrağı"),
                        ("quarter","Çeyrek"),("week","Yılın haftası")]
                render_method_group(
                    "Tarih", col, spec,
                    apply_fn=lambda keys, extra, c=col: add_date_features(df, c, keys),
                )

    if text_cols:
        st.markdown("### Metin Sütunları")
        for col in text_cols:
            with st.container(border=True):
                st.markdown(f"**Sütun:** `{col}`")
                spec = [("length","Karakter uzunluğu"),("words","Kelime sayısı"),
                        ("empty","Boş metin bayrağı"),("special","Özel karakter bayrağı"),
                        ("upper","Büyük harf oranı")]
                render_method_group(
                    "Metin", col, spec,
                    apply_fn=lambda keys, extra, c=col: add_text_features(df, c, keys),
                )

    if cat_cols:
        st.markdown("### Kategorik Sütunlar")
        for col in cat_cols:
            with st.container(border=True):
                st.markdown(f"**Sütun:** `{col}` &nbsp;·&nbsp; "
                            f"{df[col].nunique()} benzersiz değer")
                spec = [("frequency","Frekans kodlama"),("rare","Nadir kategori bayrağı"),
                        ("group_size","Grup boyutu"),("onehot","One-hot encoding")]
                render_method_group(
                    "Kategorik", col, spec,
                    apply_fn=lambda keys, extra, c=col: add_categorical_features(df, c, keys),
                )

    if num_cols:
        st.markdown("### Sayısal Sütunlar")
        for col in num_cols:
            with st.container(border=True):
                st.markdown(f"**Sütun:** `{col}` &nbsp;·&nbsp; "
                            f"ort={df[col].mean():.2f}, medyan={df[col].median():.2f}")
                spec = [("log","Log dönüşüm"),("missing_flag","Eksik değer bayrağı"),
                        ("bin","Binning (gruplara böl)"),("zscore","Z-skor (standartlaştır)"),
                        ("threshold","Eşik üstü bayrağı")]

                def extra_inputs_factory(c=col):
                    def _extra(chosen):
                        out = {}
                        if "threshold" in chosen:
                            out["threshold"] = st.number_input(
                                f"{c} için eşik değer",
                                value=float(pd.to_numeric(df[c], errors='coerce').median()),
                                key=f"thr_{c}",
                            )
                        if "bin" in chosen:
                            out["bins"] = st.slider("Bin sayısı", 2, 10, 4, key=f"bins_{c}")
                        return out
                    return _extra

                render_method_group(
                    "Sayısal", col, spec,
                    apply_fn=lambda keys, extra, c=col: add_numeric_features(
                        df, c, keys,
                        threshold=extra.get("threshold"),
                        bins=extra.get("bins", 4),
                    ),
                    extra_inputs=extra_inputs_factory(),
                )

with tab3:
    st.caption("Kendi formülünü kur: iki sütunu birleştir, bir koşul tanımla veya metin ara.")

    OP_DOCS = {
        "Oran (a/b)":   ("A sütununu B sütununa böler.",
                         "İki büyüklük arasındaki göreli ilişki önemliyse "
                         "(ör. gelir / yaş, kar / maliyet)."),
        "Fark (a-b)":   ("A'dan B'yi çıkarır.",
                         "Değişim veya marj hesaplarken (ör. fiyat - maliyet = kar)."),
        "Çarpım (a*b)": ("İki sütunu çarpar.",
                         "Toplam büyüklük elde etmek için (ör. fiyat * adet = tutar)."),
        "Toplam (a+b)": ("İki sütunu toplar.",
                         "Birleşik bir gösterge oluştururken."),
        "Flag (koşul)": ("Bir sayısal koşul sağlanıyorsa 1, değilse 0.",
                         "İş kurallarını sinyale çevirmek için (ör. gelir > 50K)."),
        "Metin içerir": ("Bir kelime metinde geçiyor mu? 1/0.",
                         "Anahtar kelime bazlı bayraklar (ör. yorum 'iade' içeriyor mu?)."),
    }

    op = st.selectbox("Operasyon", list(OP_DOCS.keys()))
    nedir, nezaman = OP_DOCS[op]
    st.info(f"**Nedir?** {nedir}\n\n**Ne zaman?** {nezaman}")

    num_cols_now = df.select_dtypes(include=[np.number]).columns.tolist()

    if op in ["Oran (a/b)", "Fark (a-b)", "Çarpım (a*b)", "Toplam (a+b)"]:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Sütun A", num_cols_now, key="ma")
        b = c2.selectbox("Sütun B", num_cols_now, key="mb")
        name = st.text_input("Yeni sütun adı", value=f"{a}_{op.split()[0].lower()}_{b}")
        if st.button("Oluştur", key="btn_manual_arith", type="primary"):
            if op.startswith("Oran"):     df[name] = df[a] / df[b].replace(0, np.nan)
            elif op.startswith("Fark"):   df[name] = df[a] - df[b]
            elif op.startswith("Çarpım"): df[name] = df[a] * df[b]
            else:                         df[name] = df[a] + df[b]
            st.session_state.df_transformed = df
            log_action(f"[manual] {name} = {a} {op} {b}")
            st.success(f"`{name}` oluşturuldu.")

    elif op == "Flag (koşul)":
        col = st.selectbox("Sütun", num_cols_now, key="fcol")
        cond = st.selectbox("Koşul", [">", ">=", "<", "<=", "=="])
        val = st.number_input("Değer", value=float(df[col].median()))
        name = st.text_input("Yeni sütun adı", value=f"{col}_flag")
        if st.button("Oluştur", key="btn_manual_flag", type="primary"):
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
        if st.button("Oluştur", key="btn_manual_text", type="primary"):
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
