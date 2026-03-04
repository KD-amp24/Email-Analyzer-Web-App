from __future__ import annotations

import re
import string
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams
from nltk import pos_tag


# =============================================================================
# Streamlit config
# =============================================================================
st.set_page_config(page_title="KWADWO'S EMAIL ANALYZER", layout="wide")
st.title("KWADWO'S EMAIL ANALYZER")
st.caption(
    "Dataset exploration • Text preprocessing • Word clouds • NLTK NLP insights "
)

# =============================================================================
# Dummy default path
# =============================================================================
DEFAULT_CSV_PATH = r"D:\Year 2 Semester 2\Info sys and proj\Phishing Email analyzer\phishing_streamlit_app\CEAS_08.csv"
RANDOM_STATE = 42


# =============================================================================
# NLTK setup (cached)
# =============================================================================
@st.cache_resource
def ensure_nltk_resources() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("sentiment/vader_lexicon", "vader_lexicon"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


# =============================================================================
# Robust CSV loading
# =============================================================================
def smart_read_csv(path_or_file) -> pd.DataFrame:
    """
    Robust CSV reader:
    - tries fast engine='c'
    - falls back to python engine with on_bad_lines skip (handles embedded newlines better)
    """
    if hasattr(path_or_file, "read"):
        try:
            df = pd.read_csv(path_or_file, sep=",", quotechar='"', engine="c", low_memory=False)
        except Exception:
            path_or_file.seek(0)
            df = pd.read_csv(path_or_file, sep=",", quotechar='"', engine="python", on_bad_lines="skip")
    else:
        path = Path(str(path_or_file))
        try:
            df = pd.read_csv(path, sep=",", quotechar='"', engine="c", low_memory=False)
        except Exception:
            df = pd.read_csv(path, sep=",", quotechar='"', engine="python", on_bad_lines="skip")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def safe_text(x) -> str:
    if isinstance(x, str):
        return x
    return "" if x is None else str(x)


# =============================================================================
# CEAS_08-like feature engineering (auto if columns exist)
# =============================================================================
def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def normalize_text(s: str) -> str:
    """
    Simple normalization for visualization/word clouds.
    - lowercase
    - strip URLs + emails
    - keep alnum + spaces
    - collapse whitespace
    """
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"http[s]?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_email_from_sender(sender: str) -> str:
    if not isinstance(sender, str):
        return ""
    m = re.search(r"<\s*([^<>@\s]+@[^<>@\s]+)\s*>", sender)
    if m:
        return m.group(1).strip().lower()
    m2 = re.search(r"\b([\w\.-]+@[\w\.-]+\.\w+)\b", sender)
    if m2:
        return m2.group(1).strip().lower()
    return ""


def extract_domain(email: str) -> str:
    if not isinstance(email, str) or "@" not in email:
        return ""
    return email.split("@", 1)[1].lower()


@st.cache_data
def try_build_ceas08_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the CSV looks like CEAS_08 (has sender/receiver/date/subject/body/label/urls),
    build engineered fields. Otherwise return df unchanged.
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    needed = {"sender", "receiver", "date", "subject", "body", "label", "urls"}
    if not needed.issubset(set(cols.keys())):
        return df

    sender = cols["sender"]
    receiver = cols["receiver"]
    date = cols["date"]
    subject = cols["subject"]
    body = cols["body"]
    label = cols["label"]
    urls = cols["urls"]

    df["date_parsed"] = safe_to_datetime(df[date])

    df[subject] = df[subject].fillna("")
    df[body] = df[body].fillna("")

    df["text_raw"] = (df[subject].astype(str) + " " + df[body].astype(str)).str.strip()
    df["text_clean"] = df["text_raw"].map(normalize_text)

    df["subject_len"] = df[subject].astype(str).str.len()
    df["body_len"] = df[body].astype(str).str.len()
    df["text_len"] = df["text_raw"].astype(str).str.len()

    df["urls_numeric"] = pd.to_numeric(df[urls], errors="coerce")
    df["has_url_in_text"] = df["text_raw"].str.contains(r"http[s]?://|www\.", regex=True, na=False).astype(int)

    df["hour_utc"] = df["date_parsed"].dt.hour
    df["weekday_utc"] = df["date_parsed"].dt.day_name()

    df["label_numeric"] = pd.to_numeric(df[label], errors="coerce").astype("Int64")

    df["sender_email"] = df[sender].map(extract_email_from_sender)
    df["sender_domain"] = df["sender_email"].map(extract_domain)

    df[receiver] = df[receiver].fillna("")
    df["receiver_domain"] = df[receiver].astype(str).str.extract(r"@(.+)$")[0].fillna("").str.lower()

    return df


# =============================================================================
# Smart default text-column selection (prevents subject-only by default)
# =============================================================================
def pick_default_text_column(df: pd.DataFrame) -> str:
    """
    Best-effort default text choice:
    1) text_raw (CEAS engineered)
    2) body (common email datasets)
    3) message/content/text/email/body/subject style columns
    4) first column
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # 1) CEAS engineered
    if "text_raw" in df.columns:
        return "text_raw"

    # 2) typical body
    if "body" in cols_lower:
        return cols_lower["body"]

    # 3) common text column names
    candidates = [
        "text", "message", "content", "email", "raw_text", "text_raw", "text_clean", "body", "subject"
    ]
    for name in candidates:
        if name in cols_lower:
            return cols_lower[name]

    # 4) fallback
    return df.columns[0]


# =============================================================================
# NLTK preprocessing
# =============================================================================
def preprocess_tokens(text: str, stop_words: set) -> List[str]:
    text = safe_text(text)
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        tok = tok.lower()
        tok = tok.strip(string.punctuation)
        if not tok:
            continue
        if tok in stop_words:
            continue
        cleaned.append(tok)
    return cleaned


@st.cache_data
def add_nltk_fields(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    ensure_nltk_resources()
    df = df.copy()

    sw = set(stopwords.words("english"))
    df["clean_tokens"] = df[text_col].apply(lambda t: preprocess_tokens(t, sw))
    df["token_len"] = df["clean_tokens"].apply(len)

    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df[text_col].apply(lambda t: float(sia.polarity_scores(safe_text(t))["compound"]))

    df["char_len"] = df[text_col].apply(lambda x: len(safe_text(x)))

    return df


# =============================================================================
# Plot helpers
# =============================================================================
def st_hist(series: pd.Series, title: str, bins: int = 40, clip_pct: Optional[float] = None):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        st.info("No data to plot.")
        return

    vals = s.values
    if clip_pct is not None:
        clip_max = np.percentile(vals, clip_pct)
        vals = np.clip(vals, 0, clip_max)

    fig, ax = plt.subplots()
    ax.hist(vals, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def st_bar_counts(counts: pd.Series, title: str, top_n: int = 20):
    if counts is None or len(counts) == 0:
        st.info("No data to plot.")
        return
    counts = counts.head(top_n)
    fig, ax = plt.subplots()
    ax.barh(list(reversed(counts.index.astype(str).tolist())), list(reversed(counts.values.tolist())))
    ax.set_title(title)
    ax.set_xlabel("Count")
    st.pyplot(fig)


def tokens_flatten(df: pd.DataFrame, token_col: str = "clean_tokens") -> List[str]:
    return [t for toks in df[token_col] for t in toks] if len(df) else []


def make_wordcloud_from_tokens(tokens: List[str], max_words: int = 200):
    if not tokens:
        st.info("No tokens available for word cloud.")
        return

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=set(ENGLISH_STOP_WORDS),
        max_words=max_words,
    ).generate(" ".join(tokens))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def top_k_tokens(tokens: List[str], k: int) -> pd.Series:
    c = Counter(tokens).most_common(k)
    return pd.Series({w: n for w, n in c}).sort_values(ascending=False)


def top_k_ngrams(tokens: List[str], n: int, k: int) -> pd.Series:
    grams = Counter([" ".join(g) for g in ngrams(tokens, n)]).most_common(k)
    return pd.Series({g: c for g, c in grams}).sort_values(ascending=False)


def tfidf_distinctive_terms(
    texts: pd.Series,
    labels: pd.Series,
    positive_label: str,
    negative_label: str,
    top_k: int = 15,
) -> Tuple[pd.Series, pd.Series]:
    """
    Distinctive TF-IDF terms per class (interpretation tool).
    Returns:
      - top terms where avg tfidf is higher in positive_label
      - top terms where avg tfidf is higher in negative_label
    """
    tmp = pd.DataFrame({"text": texts.fillna(""), "label": labels.astype(str).fillna("NA")})
    tmp = tmp[tmp["label"].isin([positive_label, negative_label])]
    if tmp["label"].nunique() < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    vec = TfidfVectorizer(max_features=25000, stop_words="english", ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(tmp["text"].astype(str).values)
    vocab = np.array(vec.get_feature_names_out())

    labs = tmp["label"].values
    pos_mask = labs == positive_label
    neg_mask = labs == negative_label

    pos_mean = np.asarray(X[pos_mask].mean(axis=0)).ravel()
    neg_mean = np.asarray(X[neg_mask].mean(axis=0)).ravel()

    diff_pos = pos_mean - neg_mean
    diff_neg = neg_mean - pos_mean

    top_pos_idx = np.argsort(diff_pos)[-top_k:][::-1]
    top_neg_idx = np.argsort(diff_neg)[-top_k:][::-1]

    pos_terms = pd.Series(diff_pos[top_pos_idx], index=vocab[top_pos_idx])
    neg_terms = pd.Series(diff_neg[top_neg_idx], index=vocab[top_neg_idx])
    return pos_terms, neg_terms


# =============================================================================
# Small upgrades helpers
# =============================================================================
def badge(text: str, kind: str = "neutral"):
    """
    Colored badge using HTML (Upgrade #1).
    kind: "danger" (phish), "success" (legit), "neutral"
    """
    styles = {
        "danger": "background:#ffe5e5;color:#8a0000;border:1px solid #ffb3b3;",
        "success": "background:#e7ffe7;color:#006b00;border:1px solid #b6ffb6;",
        "neutral": "background:#f1f3f5;color:#333;border:1px solid #d7dce0;",
    }
    st.markdown(
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;"
        f"font-weight:600;font-size:0.9rem;{styles.get(kind, styles['neutral'])}'>{text}</span>",
        unsafe_allow_html=True,
    )


def compute_risk_proxy(row_tokens: List[str], has_url: Optional[float], sentiment: Optional[float]) -> float:
    """
    Upgrade #2 (Model-confidence style visualization):
    This is NOT a classifier. It's a simple *risk proxy* score for visualization.
    - URL presence boosts risk
    - "urgency/finance/login" words boost risk
    - very negative sentiment can boost risk slightly (optional)
    Returns 0..100
    """
    urgency_words = {
        "urgent", "immediately", "action", "verify", "verification", "confirm", "suspended",
        "locked", "password", "login", "bank", "account", "invoice", "payment", "refund",
        "security", "update", "click", "link"
    }
    toks = set([t.lower() for t in row_tokens]) if isinstance(row_tokens, list) else set()
    urgency_hits = len([w for w in urgency_words if w in toks])

    score = 0.0
    # URL signal
    if has_url is not None and not np.isnan(has_url):
        score += 45.0 * float(has_url)  # 0 or 1 usually

    # urgency hits (cap)
    score += min(urgency_hits, 6) * 7.0  # up to 42

    # sentiment (optional)
    if sentiment is not None and not np.isnan(sentiment):
        if sentiment < -0.3:
            score += 8.0

    return float(np.clip(score, 0, 100))


# =============================================================================
# Sidebar UI
# =============================================================================
with st.sidebar:
    st.header("Load CSV")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    csv_path = st.text_input("…or local CSV path", value=DEFAULT_CSV_PATH)

    st.header("Performance")
    max_rows = st.slider("Max rows to load", 1000, 200000, 50000, step=5000)

    st.header("Word cloud")
    wc_max_words = st.slider("Max words", 50, 300, 200, step=10)

    st.header("Top-N")
    top_n = st.slider("Top-N tokens/bigrams/POS", 10, 50, 20, step=5)

    st.header("Columns")
    st.caption("Pick the columns your CSV uses (works for any dataset).")


# =============================================================================
# Load dataset
# =============================================================================
try:
    raw_df = smart_read_csv(uploaded if uploaded is not None else csv_path)
except Exception as e:
    st.error(f"Failed to load CSV: {e}")
    st.stop()

if len(raw_df) > max_rows:
    raw_df = raw_df.head(max_rows)

if raw_df.shape[1] == 0:
    st.error("No columns found in CSV.")
    st.stop()

# CEAS-like enhancements if available
df = try_build_ceas08_fields(raw_df)

# Column selectors
all_cols = list(df.columns)

# smart default text col (prevents subject-only by default)
default_text_col = pick_default_text_column(df)
default_text_idx = all_cols.index(default_text_col) if default_text_col in all_cols else 0

text_col = st.sidebar.selectbox("Text column (fallback)", options=all_cols, index=default_text_idx)

# Optional label column
label_choice = st.sidebar.selectbox("Label column (optional)", options=["(none)"] + all_cols, index=0)
label_col = None if label_choice == "(none)" else label_choice

# If CEAS engineered text exists, let user choose it explicitly (default to text_raw)
text_options = []
for c in ["text_raw", "text_clean"]:
    if c in df.columns:
        text_options.append(c)
if text_col not in text_options:
    text_options.append(text_col)

default_nlp_text = "text_raw" if "text_raw" in df.columns else (text_options[0] if text_options else text_col)
nlp_idx = text_options.index(default_nlp_text) if default_nlp_text in text_options else 0

chosen_text = st.sidebar.selectbox("Use this text for NLP", options=text_options, index=nlp_idx)

# Add NLTK fields
df = add_nltk_fields(df, chosen_text)

# Add risk proxy score (Upgrade #2)
# Safe-guard: if has_url_in_text doesn't exist for a dataset, treat as NaN
has_url_series = df["has_url_in_text"] if "has_url_in_text" in df.columns else pd.Series([np.nan] * len(df))
df["risk_proxy"] = [
    compute_risk_proxy(toks, has_url_series.iloc[i] if i < len(has_url_series) else np.nan, df["sentiment_score"].iloc[i])
    for i, toks in enumerate(df["clean_tokens"])
]


# =============================================================================
# Tabs
# =============================================================================
tabs = st.tabs([
    "1) Dataset Exploration",
    "2) Text Processing Display",
    "3) Word Clouds",
    "4) NLTK NLP Insights",
    "5) Phish vs Legit Comparison",
    "6) Interpretation Summary",
])


# -----------------------------------------------------------------------------
# TAB 1: Dataset Exploration
# -----------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows loaded", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("NLP text used", chosen_text)

    # Upgrade #3: Dataset summary panel
    with st.expander("Dataset summary panel (quick overview)", expanded=True):
        st.write("**Quick facts:**")
        st.write(f"- Rows: **{len(df):,}**")
        st.write(f"- Columns: **{df.shape[1]:,}**")
        st.write(f"- Selected NLP text column: **{chosen_text}**")
        if label_col:
            vc = df[label_col].astype(str).fillna("NA").value_counts(dropna=False)
            st.write(f"- Label column: **{label_col}**")
            st.write(vc.head(10))
        st.write("**Typical message size:**")
        st.write(df[["char_len", "token_len", "sentiment_score", "risk_proxy"]].describe())

    st.subheader("Preview")
    preview_cols = [chosen_text, "char_len", "token_len", "sentiment_score", "risk_proxy"]
    if label_col:
        preview_cols = [label_col] + preview_cols
    preview_cols = [c for c in preview_cols if c in df.columns]

    # FIX #1: deprecation warning removed (use width="stretch")
    st.dataframe(df[preview_cols].head(30), width="stretch")

    st.subheader("Missing values (top 15)")
    miss = df.isna().sum().sort_values(ascending=False).head(15)
    st.write(miss)

    if label_col:
        st.subheader("Label distribution")
        counts = df[label_col].astype(str).value_counts(dropna=False)
        st.write(counts)
        st.bar_chart(counts)

    st.subheader("Length distributions")
    colA, colB = st.columns(2)
    with colA:
        st_hist(df["char_len"], "Character length distribution (clipped 99th pct)", clip_pct=99)
    with colB:
        st_hist(df["token_len"], "Token count distribution (after NLTK preprocessing)")

    # URL section (only if present)
    if "has_url_in_text" in df.columns or "urls_numeric" in df.columns or "urls" in df.columns:
        st.subheader("URL indicators")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Provided URL field (overall)")
            if "urls_numeric" in df.columns:
                urls_overall = (
                    df["urls_numeric"].fillna(-1).astype(int).value_counts().sort_index()
                )
                st_bar_counts(urls_overall, "Counts of urls_numeric values", top_n=10)
            elif "urls" in df.columns:
                urls_overall = (
                    pd.to_numeric(df["urls"], errors="coerce").fillna(-1).astype(int).value_counts().sort_index()
                )
                st_bar_counts(urls_overall, "Counts of urls values", top_n=10)
            else:
                st.info("No dataset URL field detected.")

        with col2:
            st.markdown("#### Detected URL in text (has_url_in_text)")
            if "has_url_in_text" in df.columns:
                if label_col:
                    detected = (
                        df.groupby([label_col, "has_url_in_text"])
                        .size()
                        .unstack(fill_value=0)
                        .sort_index(axis=1)
                    )
                    # FIX #1: deprecation warning removed
                    st.dataframe(detected, width="stretch")
                else:
                    st.write(df["has_url_in_text"].value_counts(dropna=False))
            else:
                st.info("has_url_in_text not available for this dataset.")

    if chosen_text.lower() == "subject" or ("subject_len" in df.columns and df["char_len"].median() < 80):
        st.warning(
            "You appear to be analyzing a short text field (e.g., subject-only). "
            "For richer NLP insights, switch 'Use this text for NLP' to text_raw/body if available."
        )

    st.markdown("### Interpretation")
    st.info(
        "This section sets the stage for your NLP results: "
        "class balance (if labels exist), missingness, and length patterns. "
        "These often explain why phishing is detectable even before training a classifier."
    )


# -----------------------------------------------------------------------------
# TAB 2: Text Processing Display
# -----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Pick a row and show text → preprocessing → tokens")

    idx = st.number_input("Row index", min_value=0, max_value=int(len(df) - 1), value=0, step=1)
    row = df.iloc[int(idx)]

    if label_col:
        st.write(f"**Label:** `{row[label_col]}`")

    if "has_url_in_text" in df.columns:
        st.write(f"**Detected URL in text:** `{row.get('has_url_in_text')}`")

    st.write(f"**Risk proxy (0–100):** `{row.get('risk_proxy', np.nan):.1f}`")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Original text (selected NLP field)")
        st.text_area("Original", safe_text(row[chosen_text]), height=260)
    with colB:
        st.markdown("#### Clean tokens (NLTK)")
        st.text_area("Tokens", " ".join(row["clean_tokens"]), height=260)

    st.markdown("### Interpretation")
    st.success(
        "This panel proves your pipeline is doing real NLP preprocessing:\n\n"
        "- Tokenization breaks text into words.\n"
        "- Lowercasing and punctuation stripping reduce formatting noise.\n"
        "- Stopword removal focuses analysis on meaning-bearing terms."
    )


# -----------------------------------------------------------------------------
# TAB 3: Word Clouds
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Word clouds (visual summary)")
    st.caption("Word clouds are clues. Use token/bigram counts for evidence (next tabs).")

    all_tokens = tokens_flatten(df, "clean_tokens")
    st.markdown("#### Overall")
    make_wordcloud_from_tokens(all_tokens, max_words=wc_max_words)

    if label_col:
        st.divider()
        st.markdown("#### By label (first 6 labels)")
        labels = sorted(df[label_col].astype(str).fillna("NA").unique().tolist())
        for lab in labels[:6]:
            st.markdown(f"**Label = {lab}**")
            seg = df[df[label_col].astype(str).fillna("NA") == lab]
            seg_tokens = tokens_flatten(seg, "clean_tokens")
            make_wordcloud_from_tokens(seg_tokens, max_words=wc_max_words)


# -----------------------------------------------------------------------------
# TAB 4: NLTK NLP Insights
# -----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("NLTK NLP insights (evidence-based)")

    try:
        if label_col:
            segment = st.selectbox("Segment", ["All"] + sorted(df[label_col].astype(str).fillna("NA").unique().tolist()))
            seg_df = df if segment == "All" else df[df[label_col].astype(str).fillna("NA") == segment]
        else:
            seg_df = df
            segment = "All"

        seg_tokens = tokens_flatten(seg_df, "clean_tokens")

        if not seg_tokens:
            st.info("No tokens available in this segment.")
        else:
            st.markdown("### Top tokens")
            st_bar_counts(top_k_tokens(seg_tokens, top_n), f"Top {top_n} tokens — {segment}", top_n=top_n)

            st.markdown("### Top bigrams (2-word phrases)")
            st_bar_counts(top_k_ngrams(seg_tokens, 2, top_n), f"Top {top_n} bigrams — {segment}", top_n=top_n)

            st.markdown("### POS tag distribution")
            tagged = pos_tag(seg_tokens)
            pos_counts = Counter([tag for _, tag in tagged]).most_common(min(top_n, 15))
            pos_series = pd.Series({k: v for k, v in pos_counts}).sort_values(ascending=False)
            st_bar_counts(pos_series, f"Top POS tags — {segment}", top_n=min(top_n, 15))

            st.markdown("### Sentiment (VADER compound)")
            st_hist(seg_df["sentiment_score"], f"Sentiment distribution — {segment}", bins=40)
            st.write(seg_df["sentiment_score"].describe())

            st.markdown("### Risk proxy distribution (0–100)")
            st_hist(seg_df["risk_proxy"], f"Risk proxy distribution — {segment}", bins=40)
            st.write(seg_df["risk_proxy"].describe())

        st.markdown("### Interpretation tips")
        st.success(
            "- Tokens show dominant vocabulary after cleaning.\n"
            "- Bigrams reveal intent (stronger evidence than single words).\n"
            "- POS tags describe writing style (directive verbs, noun-heavy content).\n"
            "- Sentiment helps explain emotional manipulation; phishing can be urgent without being negative.\n"
            "- Risk proxy is a simple visualization score (not a trained model)."
        )
    except Exception as e:
        st.error("Something failed while generating the NLP insights panel.")
        st.exception(e)


# -----------------------------------------------------------------------------
# TAB 5: Phish vs Legit Comparison
# -----------------------------------------------------------------------------
with tabs[4]:
    try:
        st.subheader("Phish vs Legit Comparison")

        if not label_col:
            st.warning("No label column selected. Go to the sidebar → pick your label column (phish vs legit).")
            st.stop()

        labels = sorted(df[label_col].astype(str).fillna("NA").unique().tolist())

        if len(labels) < 2:
            st.warning("Your label column has fewer than 2 unique values. Pick a different label column.")
            st.stop()

        st.markdown("### Select labels")

        default_pos = "1" if "1" in labels else labels[0]

        colx, coly = st.columns(2)
        with colx:
            pos_label = st.selectbox(
                "Which label represents phishing/spam?",
                options=labels,
                index=labels.index(default_pos) if default_pos in labels else 0,
                key="tab5_pos_label",  # unique key
            )

        neg_candidates = [l for l in labels if l != pos_label]
        with coly:
            neg_label = st.selectbox(
                "Which label represents legitimate/safe?",
                options=neg_candidates,
                index=0,
                key="tab5_neg_label",  # unique key
            )

        st.markdown("### Status badges")
        badge(f"PHISHING = {pos_label}", kind="danger")
        st.write("")
        badge(f"LEGIT = {neg_label}", kind="success")

        df_pos = df[df[label_col].astype(str).fillna("NA") == pos_label]
        df_neg = df[df[label_col].astype(str).fillna("NA") == neg_label]

        st.markdown("### Quick comparison stats")
        a, b, c, d = st.columns(4)
        a.metric("Phishing rows", f"{len(df_pos):,}")
        b.metric("Legit rows", f"{len(df_neg):,}")
        c.metric("Avg risk (phish)", f"{df_pos['risk_proxy'].mean():.1f}" if len(df_pos) else "N/A")
        d.metric("Avg risk (legit)", f"{df_neg['risk_proxy'].mean():.1f}" if len(df_neg) else "N/A")

        if "has_url_in_text" in df.columns:
            rate_pos = df_pos["has_url_in_text"].mean() if len(df_pos) else np.nan
            rate_neg = df_neg["has_url_in_text"].mean() if len(df_neg) else np.nan
            c1, c2 = st.columns(2)
            c1.metric(f"URL rate (phish={pos_label})", f"{rate_pos*100:.1f}%" if not np.isnan(rate_pos) else "N/A")
            c2.metric(f"URL rate (legit={neg_label})", f"{rate_neg*100:.1f}%" if not np.isnan(rate_neg) else "N/A")

        st.markdown("### Risk proxy comparison (visual)")
        risk_summary = pd.DataFrame({
            "Class": [f"Phish ({pos_label})", f"Legit ({neg_label})"],
            "Avg risk proxy": [
                float(df_pos["risk_proxy"].mean()) if len(df_pos) else np.nan,
                float(df_neg["risk_proxy"].mean()) if len(df_neg) else np.nan,
            ],
            "Median risk proxy": [
                float(df_pos["risk_proxy"].median()) if len(df_pos) else np.nan,
                float(df_neg["risk_proxy"].median()) if len(df_neg) else np.nan,
            ],
        }).set_index("Class")

        st.bar_chart(risk_summary)

        st.divider()
        st.markdown("### Vocabulary + intent comparison")

        pos_tokens = tokens_flatten(df_pos, "clean_tokens")
        neg_tokens = tokens_flatten(df_neg, "clean_tokens")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("#### Top tokens (phish)")
            st_bar_counts(top_k_tokens(pos_tokens, top_n), f"Top {top_n} tokens — phish", top_n=top_n)
            st.markdown("#### Top bigrams (phish)")
            st_bar_counts(top_k_ngrams(pos_tokens, 2, top_n), f"Top {top_n} bigrams — phish", top_n=top_n)

        with colB:
            st.markdown("#### Top tokens (legit)")
            st_bar_counts(top_k_tokens(neg_tokens, top_n), f"Top {top_n} tokens — legit", top_n=top_n)
            st.markdown("#### Top bigrams (legit)")
            st_bar_counts(top_k_ngrams(neg_tokens, 2, top_n), f"Top {top_n} bigrams — legit", top_n=top_n)

        st.divider()
        st.markdown("### Distinctive terms (TF-IDF difference)")

        pos_terms, neg_terms = tfidf_distinctive_terms(
            df[chosen_text],
            df[label_col],
            positive_label=pos_label,
            negative_label=neg_label,
            top_k=15,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**More distinctive for phishing ({pos_label})**")
            st.write(pos_terms if len(pos_terms) else "Not enough data to compute TF-IDF distinct terms.")
        with c2:
            st.markdown(f"**More distinctive for legitimate ({neg_label})**")
            st.write(neg_terms if len(neg_terms) else "Not enough data to compute TF-IDF distinct terms.")

    except Exception as e:
        st.error("Tab 5 crashed. Here is the full error:")
        st.exception(e)


# -----------------------------------------------------------------------------
# TAB 6: Interpretation Summary
# -----------------------------------------------------------------------------
with tabs[5]:
    try:
        st.subheader("Interpretation summary")

        st.markdown("### What the app did (high-level)")
        st.write(f"- CSV loaded: **{len(df):,} rows**")
        st.write(f"- NLP text used: **{chosen_text}**")
        st.write(f"- Columns available: **{df.shape[1]:,}**")

        st.markdown("### Key numeric signals (always available)")
        cols_needed = ["char_len", "token_len", "sentiment_score", "risk_proxy"]
        cols_present = [c for c in cols_needed if c in df.columns]

        if not cols_present:
            st.warning("None of the summary numeric columns are present in df (unexpected).")
        else:
            st.write(df[cols_present].describe())

        if label_col:
            st.markdown("### Label distribution")
            st.write(f"- Label column: **{label_col}**")
            st.write(df[label_col].astype(str).fillna("NA").value_counts(dropna=False))
        else:
            st.info("Tip: Pick a label column in the sidebar to enable class summaries.")

        st.markdown("### What we did:")
        st.info(
            "We performed exploratory analysis to understand the dataset structure (missingness, class balance, "
            "and message length). We then applied NLTK-based preprocessing (tokenization, lowercasing, punctuation removal, "
            "and stopword filtering) to isolate meaning-bearing terms. Word clouds provided an initial visual summary, "
            "while token and bigram frequency offered evidence-backed patterns that indicate suspicious intent. "
            "POS tagging and VADER sentiment were used to describe writing style and emotional tone often associated "
            "with social engineering. Finally, we included a simple risk proxy score (based on URLs and urgency language) "
            "as a visual confidence-style indicator to compare phishing vs legitimate messages."
        )

        st.markdown("### Optional export")
        if st.button("Prepare processed CSV for download", key="tab6_export_btn"):  # unique key
            out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download processed_dataset.csv",
                data=out,
                file_name="processed_dataset.csv",
                mime="text/csv",
                key="tab6_download_btn",  # unique key
            )

    except Exception as e:
        st.error("Tab 6 crashed. Here is the full error:")
        st.exception(e)