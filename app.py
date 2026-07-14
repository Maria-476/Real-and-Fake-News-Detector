import streamlit as st
import joblib
import time

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# GLOBAL STYLES
# ---------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root {
    --primary: #2563EB;
    --secondary: #4F46E5;
    --success: #10B981;
    --warning: #F59E0B;
    --fake: #DC2626;
    --bg: #F8FAFC;
    --text: #1E293B;
}
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}
.stApp {
    background: radial-gradient(circle at 20% 0%, rgba(37,99,235,0.06) 0%, rgba(248,250,252,1) 45%),
                radial-gradient(circle at 80% 20%, rgba(79,70,229,0.05) 0%, rgba(248,250,252,1) 50%),
                #F8FAFC;
}
.block-container {
    max-width: 1150px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}
/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    animation: fadeIn 0.5s ease-in-out;
}
.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(37,99,235,0.1), rgba(79,70,229,0.1));
    color: var(--primary);
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.3px;
    margin-bottom: 18px;
    border: 1px solid rgba(37,99,235,0.15);
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1.1rem;
    color: #64748B;
    font-weight: 400;
    max-width: 550px;
    margin: 0 auto;
}
/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 8px 32px rgba(30,41,59,0.06);
    padding: 2rem;
    margin-bottom: 1.5rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeIn 0.5s ease-in-out;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(30,41,59,0.09);
}
/* Text area */
.stTextArea textarea {
    border-radius: 16px !important;
    border: 1.5px solid #E2E8F0 !important;
    padding: 16px !important;
    font-size: 15px !important;
    background: rgba(255,255,255,0.8) !important;
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    color: white;
    font-weight: 700;
    font-size: 16px;
    border: none;
    border-radius: 14px;
    padding: 0.85rem 1.5rem;
    box-shadow: 0 6px 20px rgba(37,99,235,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stButton > button:hover {
    transform: scale(1.015) translateY(-1px);
    box-shadow: 0 10px 28px rgba(37,99,235,0.4);
    color: white;
}
.stButton > button:active {
    transform: scale(0.99);
}
/* Prediction cards */
.pred-card {
    border-radius: 22px;
    padding: 2rem 2.2rem;
    color: white;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.4s ease-in-out;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}
.pred-card.real {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
}
.pred-card.fake {
    background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
}
.pred-title {
    font-size: 1.7rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.pred-sub {
    opacity: 0.92;
    font-size: 0.98rem;
    margin-bottom: 18px;
}
.confidence-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 6px;
    opacity: 0.95;
}
.confidence-track {
    width: 100%;
    height: 12px;
    background: rgba(255,255,255,0.28);
    border-radius: 999px;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    background: rgba(255,255,255,0.95);
    border-radius: 999px;
    animation: growBar 0.6s ease-out;
}
@keyframes growBar {
    from { width: 0%; }
}
.explanation-box {
    margin-top: 18px;
    background: rgba(255,255,255,0.15);
    border-radius: 14px;
    padding: 14px 18px;
    font-size: 0.92rem;
    line-height: 1.5;
}
/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.6);
    padding: 1.3rem 1rem;
    text-align: center;
    box-shadow: 0 6px 20px rgba(30,41,59,0.05);
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--primary);
}
.metric-label {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 500;
    margin-top: 2px;
}
/* Section heading */
.section-heading {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
/* Footer */
.footer {
    text-align: center;
    color: #94A3B8;
    font-size: 0.85rem;
    padding: 2rem 0 1rem 0;
    border-top: 1px solid #E2E8F0;
    margin-top: 2rem;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
/* Hide default streamlit chrome (leave header untouched so sidebar toggle keeps working) */
#MainMenu, footer {visibility: hidden;}
/* Force readable text everywhere, regardless of light/dark theme */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: var(--text) !important;
}
.stTextArea textarea {
    color: var(--text) !important;
}
.stTextArea textarea::placeholder {
    color: #94A3B8 !important;
}
/* Expander styling - glass card look with readable text */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.75) !important;
    backdrop-filter: blur(10px);
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
    box-shadow: 0 6px 20px rgba(30,41,59,0.05);
    margin-bottom: 1rem;
}
[data-testid="stExpander"] summary {
    color: var(--text) !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] p, [data-testid="stExpander"] li, [data-testid="stExpander"] strong {
    color: var(--text) !important;
}
/* Input card (native bordered container) styling */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    border-radius: 22px !important;
    box-shadow: 0 8px 32px rgba(30,41,59,0.06);
}
/* Sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("Real&Fake_news.pkl")

model = load_model()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 About")
    st.markdown(
        "This app detects fake vs. real news articles using a "
        "**TF-IDF + Linear SVM** model trained on ~45,700 labeled articles."
    )
    st.markdown("---")
    st.markdown("### 📊 Model")
    st.markdown("- **Accuracy:** 97.66%\n- **Algorithm:** LinearSVC\n- **Features:** TF-IDF (1000 features)")
    st.markdown("---")
    st.markdown("### ⚠️ Limitations")
    st.markdown("Trained on 2015–2017 data — may be less reliable on very recent topics.")
    st.markdown("---")
    st.markdown("### 👤 Author")
    st.markdown("Maria Anwar")

# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">🤖 AI-Powered Verification</div>
    <h1>Fake News Detector</h1>
    <p>Paste any news article below and let a machine learning model assess whether it reads as real or fake — powered by TF-IDF and a Support Vector Machine.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------
st.markdown("""
<div class="glass-card" style="padding: 1rem 1.5rem; margin-bottom: 1rem;">
    <span style="font-weight:600; color:#F59E0B;">⚠️ Note:</span>
    <span style="color:#475569; font-size:0.92rem;"> This model was trained on 2015–2017 news data. It may be less reliable on very recent topics or unfamiliar vocabulary — see "How It Works" below for details.</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INPUT CARD
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="section-heading">📰 Paste Article Text</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "Article text",
        height=220,
        placeholder="Paste the full news article text here...",
        label_visibility="collapsed",
    )

    analyze = st.button("🔍  Analyze News", use_container_width=True)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if analyze:
    if not user_input.strip():
        st.warning("Please paste some article text before analyzing.")
    else:
        with st.spinner("Analyzing article..."):
            time.sleep(0.6)  # brief pause for perceived processing
            start = time.time()
            prediction = model.predict([user_input])[0]
            score = model.decision_function([user_input])[0]
            elapsed = (time.time() - start) * 1000

        # convert decision_function score to a rough 0-100% confidence display
        confidence_pct = min(99, max(55, int(50 + abs(score) * 15)))

        if prediction == 1:
            st.markdown(f"""
            <div class="pred-card real">
                <div class="pred-title">✅ REAL NEWS</div>
                <div class="pred-sub">This article's writing style closely matches patterns typical of legitimate news reporting.</div>
                <div class="confidence-label"><span>📊 Confidence</span><span>{confidence_pct}%</span></div>
                <div class="confidence-track"><div class="confidence-fill" style="width:{confidence_pct}%;"></div></div>
                <div class="explanation-box">The model found linguistic patterns consistent with credible reporting (neutral tone, factual structure). Always verify important claims with a trusted source.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-card fake">
                <div class="pred-title">❌ LIKELY FAKE</div>
                <div class="pred-sub">This article's writing style matches patterns commonly seen in misleading or fabricated news.</div>
                <div class="confidence-label"><span>📊 Confidence</span><span>{confidence_pct}%</span></div>
                <div class="confidence-track"><div class="confidence-fill" style="width:{confidence_pct}%;"></div></div>
                <div class="explanation-box">⚠️ Caution: the model detected tone/style patterns associated with misinformation. Cross-check with a fact-checking source before sharing.</div>
            </div>
            """, unsafe_allow_html=True)

        # stats row for this prediction
        st.markdown('<div class="section-heading">📈 Prediction Details</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{confidence_pct}%</div><div class="metric-label">Confidence Score</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">{elapsed:.0f}ms</div><div class="metric-label">Processing Time</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value">SVM</div><div class="metric-label">Model Used</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# MODEL STATISTICS
# ---------------------------------------------------------
st.markdown('<div class="section-heading">📊 Model Statistics</div>', unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)
with s1:
    st.markdown('<div class="metric-card"><div class="metric-value">97.66%</div><div class="metric-label">Test Accuracy</div></div>', unsafe_allow_html=True)
with s2:
    st.markdown('<div class="metric-card"><div class="metric-value">45,757</div><div class="metric-label">Training Articles</div></div>', unsafe_allow_html=True)
with s3:
    st.markdown('<div class="metric-card"><div class="metric-value">TF-IDF + LinearSVC</div><div class="metric-label">Approach</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# HOW IT WORKS
# ---------------------------------------------------------
with st.expander("🤖  How It Works"):
    st.markdown("""
    This model uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert article text into
    numerical features based on word importance, then a **Linear Support Vector Classifier (SVM)** draws
    a boundary between patterns typical of real vs. fake news.

    **Training data:** ~45,700 labeled articles (2015–2017 era).

    **Known limitations:**
    - Trained on 2015–2017 vocabulary - may be less confident on very recent topics/events
    - Detects *writing style* patterns (tone, structure) rather than verifying facts directly
    - A well-written fabricated article can occasionally be misclassified as real
    """)

with st.expander("💡  Tips for Identifying Fake News Yourself"):
    st.markdown("""
    - Check the source - is it a known, reputable outlet?
    - Look for emotionally charged or sensational language
    - Verify claims against multiple independent sources
    - Check the publish date — old stories are sometimes recirculated as new
    - Be wary of articles with no named author or citations
    """)

with st.expander("ℹ️  About This Model"):
    st.markdown("""
    Built as part of an ML portfolio project exploring text classification with Support Vector Machines.
    Includes a documented data-leakage investigation and honest evaluation of real-world generalization
    limitations. Full write-up available in the project README.
    """)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    Made with Streamlit & Scikit-learn · Fake News Detector Portfolio Project
</div>
""", unsafe_allow_html=True)