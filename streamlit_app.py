"""
support_ticket_app.py
=====================
Production-Grade Streamlit Web Application:
Support Ticket Classification & Prioritization Decision-Support System.
Runs locally at: http://localhost:8501
"""

import os
import re
import random
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------------------------------------------------------
# 1. Page Configuration & Theme
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Support Ticket Intelligence | NLP Decision-Support",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark / Light Adaptive CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 12px;
        padding: 22px 28px;
        color: #ffffff;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .result-card {
        background-color: #1e293b;
        color: #ffffff;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #334155;
        margin-top: 16px;
        margin-bottom: 20px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .badge-high {
        background-color: #ef4444;
        color: #ffffff;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }
    
    .badge-medium {
        background-color: #f59e0b;
        color: #ffffff;
    }
    
    .badge-low {
        background-color: #10b981;
        color: #ffffff;
    }
    
    .badge-cat {
        background-color: #3b82f6;
        color: #ffffff;
    }
    
    .sla-alert-box {
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
        color: #fca5a5;
    }
    
    .sla-safe-box {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 12px 0 20px 0;
        color: #6ee7b7;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Setup NLTK Dependencies Quietly
# -------------------------------------------------------------
for resource in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Production NLP text cleaner and lemmatizer."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' <EMAIL> ', text)
    text = re.sub(r'\$\s*\d+(?:,\d{3})*(?:\.\d+)?', ' <CURRENCY> ', text)
    text = re.sub(r'\b\d+\b', ' <NUM> ', text)
    text = re.sub(r'[^a-z0-9\s<>]', ' ', text)
    tokens = text.split()
    
    cleaned = []
    entity_tags = {'<currency>', '<num>', '<url>', '<email>'}
    for t in tokens:
        if t in stop_words and t not in entity_tags:
            continue
        if len(t) <= 1 and t not in entity_tags:
            continue
        lemma = lemmatizer.lemmatize(t, pos='v')
        lemma = lemmatizer.lemmatize(lemma, pos='n')
        cleaned.append(lemma)
    return " ".join(cleaned)

# Critical SLA Regex Safeguards
CRITICAL_SLA_PATTERNS = [
    r'\bproduction (?:down|offline|halted|unavailable)\b',
    r'\b(?:security|data) breach\b',
    r'\bunauthorized (?:charge|access|transaction)\b',
    r'\b(?:500|502|503|504) internal server error\b',
    r'\bhipaa audit\b',
    r'\bgdpr (?:penalty|breach|deletion|dpa)\b',
    r'\b2fa (?:lost|lockout|fail)\b',
    r'\bdatabase connection timeout\b'
]
compiled_guardrails = [re.compile(p, re.IGNORECASE) for p in CRITICAL_SLA_PATTERNS]

# -------------------------------------------------------------
# 3. Model Initialization (Cached)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Training NLP Models & Initializing Decision Engines...")
def initialize_system():
    random.seed(42)
    np.random.seed(42)

    billing_tmpls = [
        ("Unauthorized credit card charge of ${amt}", "I noticed an unauthorized charge of ${amt} on card ending {card} on {date}. We need immediate cancellation and refund.", "High"),
        ("Payment gateway failed - subscription suspended", "Our payment failed due to gateway timeout. Our production access is blocked. Please reactivate our subscription immediately.", "High"),
        ("Double charged for annual enterprise license", "We were billed twice (${amt} x 2) for our annual plan {inv}. This has overdrafted our operating account. Please refund ASAP.", "High"),
        ("Request copy of VAT invoice for fiscal year", "Please provide tax-compliant PDF receipt and VAT breakdown for invoice {inv} for our finance audit.", "Medium"),
        ("Refund request for unused team seats", "We downgraded from 50 to 30 seats last week. Can we receive a prorated refund or account credit for next cycle?", "Medium"),
        ("Discrepancy in monthly usage charges", "Our invoice {inv} shows 12,000 API calls, but our logs record only 8,000. Could you review our billing breakdown?", "Medium"),
        ("Update company billing address", "Please update our billing address on future receipts to 100 Main Street, Suite 400, San Francisco CA.", "Low"),
        ("Inquiry regarding annual vs monthly discount", "Could you explain the pricing discount if our team switches from monthly billing to annual commitment?", "Low")
    ]

    tech_tmpls = [
        ("CRITICAL: Production API 500 internal server error", "All requests to endpoint /v1/checkout are returning HTTP 500 error since 10:00 AM UTC. Customers cannot complete orders.", "High"),
        ("Database connection timeout in US-East cluster", "Our production database pool is dropping connections with timeout errors. Application is completely offline.", "High"),
        ("Mobile app crashing on launch after v4.2 update", "Both iOS and Android builds crash immediately upon launch. Severe crash rate reported by over 500 users.", "High"),
        ("Webhook delivery failing intermittently", "Webhook notifications to our endpoint https://api.mycompany.com/webhook are timing out with code 504 on 15% of events.", "Medium"),
        ("CSV data export job stuck in pending state", "Our scheduled export job {id} has been pending for over 6 hours without finishing. Please restart or release lock.", "Medium"),
        ("Dashboard analytics widget not refreshing", "The real-time chart on the primary analytics dashboard stopped refreshing metric counts after yesterday maintenance.", "Medium"),
        ("Minor CSS styling issue on settings tab", "The toggle button on the dark theme settings menu overlaps slightly with the notification label in Firefox.", "Low"),
        ("Tooltip text truncated on user profile", "In the user settings page, the hover tooltip for session timeout is truncated on screen resolutions below 1080p.", "Low")
    ]

    account_tmpls = [
        ("SECURITY ALERT: Admin locked out during suspicious login attempt", "Our primary administrator account {email} has been locked out after 3 failed login alerts from unknown IP. Need urgent unlock.", "High"),
        ("2FA token lost and emergency recovery codes expired", "Our security officer lost their hardware 2FA device and backup codes fail. We cannot access our corporate tenant.", "High"),
        ("SAML SSO integration failing for entire organization", "Our Okta SSO configuration broke after identity provider cert rotation. 300 employees are currently unable to sign in.", "High"),
        ("Cannot transfer ownership of team workspace", "Trying to transfer primary owner permissions to user {email}, but the dropdown confirms error 403 forbidden.", "Medium"),
        ("Invitation email not arriving for new team member", "We invited candidate {email} to our team space, but verification emails are not arriving in inbox or spam folder.", "Medium"),
        ("Role-based access permissions not applying correctly", "User {email} has been assigned Analyst role, but they still have full Editor write access to production projects.", "Medium"),
        ("How to change profile avatar and display name", "Could you guide me on where in settings I can update my display name and profile picture?", "Low"),
        ("Request to change registered email address", "I would like to change my account login email from oldcorp.com to my new email domain newcorp.com.", "Low")
    ]

    general_tmpls = [
        ("Urgent compliance inquiry for impending HIPAA audit", "We have a regulatory HIPAA compliance audit tomorrow morning. We urgently need your SOC2 Type II report and signed BAA agreement.", "High"),
        ("Legal inquiry regarding GDPR data processing addendum", "Our legal counsel requires counter-signed DPA and data residency confirmation within 24 hours to avoid regulatory penalty.", "High"),
        ("Question regarding enterprise API rate limits", "What are the tier limits for concurrent requests under the Enterprise SLA? Can we burst up to 10,000 req/min during peak?", "Medium"),
        ("Supported integration with Snowflake data warehouse", "Do you offer a native connector for Snowflake or is reverse ETL via Fivetran/Segment recommended?", "Medium"),
        ("Feature request: native dark mode for desktop client", "Would love to see a native dark mode theme option in the upcoming desktop app release roadmap.", "Low"),
        ("Documentation clarification on webhook retry policy", "Can you clarify whether webhook retries follow exponential backoff and what the maximum retry count is?", "Low"),
        ("Inquiry on standard product roadmap and Q4 releases", "Where can we find public information on features planned for release in the upcoming Q4 quarter?", "Low"),
        ("Office hours and customer support availability on holidays", "Could you provide the support team operating schedule for the upcoming New Year holiday period?", "Low")
    ]

    categories_catalog = [
        ("Billing", billing_tmpls),
        ("Technical Issue", tech_tmpls),
        ("Account", account_tmpls),
        ("General Query", general_tmpls)
    ]

    rows = []
    tid = 1001
    for cat_name, tmpls in categories_catalog:
        for i in range(50):
            subj_tmpl, body_tmpl, pri_val = tmpls[i % len(tmpls)]
            amt = random.choice([49, 99, 199, 450, 1200, 3500])
            card = random.choice(["4312", "9821", "1094", "6734", "8820"])
            date = random.choice(["yesterday", "Oct 14", "this morning", "Monday"])
            inv = f"INV-{random.randint(10000, 99999)}"
            job_id = f"EXP-{random.randint(1000, 9999)}"
            email = f"user_{random.randint(100, 999)}@company.org"
            
            s = subj_tmpl.format(amt=amt, card=card, date=date, inv=inv, id=job_id, email=email)
            b = body_tmpl.format(amt=amt, card=card, date=date, inv=inv, id=job_id, email=email)
            rows.append({
                "ticket_id": f"TICK-{tid}",
                "ticket_subject": s,
                "ticket_body": b,
                "category": cat_name,
                "priority": pri_val
            })
            tid += 1

    df = pd.DataFrame(rows)
    df['raw_text'] = df['ticket_subject'] + " " + df['ticket_body']
    df['clean_text'] = df['raw_text'].apply(preprocess_text)

    # Train/Test Split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        df['clean_text'], df['category'], test_size=0.25, random_state=42, stratify=df['category']
    )
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        df['clean_text'], df['priority'], test_size=0.25, random_state=42, stratify=df['priority']
    )

    tfidf_params = {'ngram_range': (1, 2), 'max_features': 3000, 'sublinear_tf': True, 'min_df': 2}
    
    # Category Pipeline
    cat_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])
    cat_pipe.fit(X_train_c, y_train_c)

    # Priority Pipeline with Balanced Weights
    pri_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('clf', LogisticRegression(class_weight='balanced', C=1.2, max_iter=1000, random_state=42))
    ])
    pri_pipe.fit(X_train_p, y_train_p)

    def extract_top_n_grams(pipeline, top_k=10):
        vec = pipeline.named_steps['tfidf']
        clf = pipeline.named_steps['clf']
        f_names = np.array(vec.get_feature_names_out())
        results = {}
        for idx, cls in enumerate(clf.classes_):
            top_idx = np.argsort(clf.coef_[idx])[-top_k:][::-1]
            results[cls] = list(zip(f_names[top_idx], clf.coef_[idx][top_idx]))
        return results

    cat_top_kw = extract_top_n_grams(cat_pipe, top_k=10)
    pri_top_kw = extract_top_n_grams(pri_pipe, top_k=10)

    cat_acc = cat_pipe.score(X_test_c, y_test_c)
    pri_acc = pri_pipe.score(X_test_p, y_test_p)
    cat_cm = confusion_matrix(y_test_c, cat_pipe.predict(X_test_c), labels=sorted(df['category'].unique()))
    pri_cm = confusion_matrix(y_test_p, pri_pipe.predict(X_test_p), labels=['Low', 'Medium', 'High'])

    return {
        'df': df,
        'cat_pipe': cat_pipe,
        'pri_pipe': pri_pipe,
        'cat_acc': cat_acc,
        'pri_acc': pri_acc,
        'cat_cm': cat_cm,
        'pri_cm': pri_cm,
        'cat_classes': sorted(df['category'].unique()),
        'pri_classes': ['Low', 'Medium', 'High'],
        'cat_top_kw': cat_top_kw,
        'pri_top_kw': pri_top_kw
    }

system = initialize_system()

# -------------------------------------------------------------
# 4. Header Hero
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-size: 0.8rem; font-weight: 700; background: rgba(255,255,255,0.2); padding: 3px 10px; border-radius: 12px;">
            ENTERPRISE NLP TRIAGE • ACTIVE LOCALHOST:8501
        </span>
        <span style="font-size: 0.85rem; color: #cbd5e1;">● System Operational</span>
    </div>
    <h1 style="margin: 0; font-size: 2rem; font-weight: 800; color: #ffffff;">
        Support Ticket Classification & Prioritization
    </h1>
    <p style="margin: 6px 0 0 0; font-size: 1rem; color: #cbd5e1;">
        Automated multi-class topic routing, class-balanced urgency triage, and zero-SLA-breach guardrails.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 5. Preset Data Dictionary
# -------------------------------------------------------------
PRESETS = {
    "🔴 Critical Outage (P0)": {
        "subj": "CRITICAL: Database connection timeout in production cluster",
        "body": "All API endpoints are returning HTTP 500 error since 10:00 AM UTC. Customer checkout is completely offline!"
    },
    "💳 Billing Fraud / Unauthorized": {
        "subj": "Unauthorized charge on Visa corporate card ending 9821",
        "body": "We noticed an unauthorized charge of $1,250 on our statement for invoice INV-48211. Reverse this charge immediately."
    },
    "🔐 2FA / SSO Lockout": {
        "subj": "SAML SSO integration failing for entire organization",
        "body": "Our Okta certificate expired this morning. 300 employees are completely locked out of the company dashboard."
    },
    "💡 Feature Request (Low)": {
        "subj": "Feature request: native dark mode for desktop client",
        "body": "Could the design team please consider adding a dark mode toggle option in the upcoming Q4 product roadmap?"
    },
    "⚖️ GDPR / HIPAA Audit": {
        "subj": "Urgent compliance inquiry for impending HIPAA audit",
        "body": "We have an official regulatory audit tomorrow morning. We urgently need your signed BAA and SOC2 Type II report."
    }
}

# -------------------------------------------------------------
# 6. Sidebar Controls
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Supervisor Settings")
    confidence_threshold = st.slider(
        "Human Review Confidence Threshold",
        min_value=0.30,
        max_value=0.85,
        value=0.55,
        step=0.05,
        help="Flag tickets where model confidence is below this cutoff."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Operational Health")
    st.metric("Category Model Accuracy", f"{system['cat_acc'] * 100:.1f}%")
    st.metric("Priority Model Accuracy", f"{system['pri_acc'] * 100:.1f}%")
    st.metric("Training Dataset", f"{len(system['df'])} tickets")
    
    st.markdown("---")
    st.caption("Machine Learning Task 2 • Future Interns")

# -------------------------------------------------------------
# 7. Navigation Tabs
# -------------------------------------------------------------
tab_triage, tab_batch, tab_analytics, tab_explain, tab_roi = st.tabs([
    "🎯 Real-Time Triage Studio",
    "📂 Batch CSV Processing",
    "📊 Model Performance & Matrices",
    "🔍 Linguistic Explainability",
    "💼 Executive ROI & SLA Impact"
])

# Inference Engine
def execute_prediction(subject: str, body: str):
    raw = f"{subject or ''} {body or ''}".strip()
    clean = preprocess_text(raw)
    
    cat_pred = system['cat_pipe'].predict([clean])[0]
    cat_probas = system['cat_pipe'].predict_proba([clean])[0]
    cat_conf = float(np.max(cat_probas))
    
    pri_pred = system['pri_pipe'].predict([clean])[0]
    pri_probas = system['pri_pipe'].predict_proba([clean])[0]
    pri_conf = float(np.max(pri_probas))
    
    # SLA Check
    sla_triggered = False
    matched_rule = None
    for pattern in compiled_guardrails:
        match = pattern.search(raw)
        if match:
            sla_triggered = True
            matched_rule = match.group(0)
            break
            
    final_pri = "High" if sla_triggered else pri_pred
    human_flag = (cat_conf < confidence_threshold) or (pri_conf < confidence_threshold) or sla_triggered
    
    queue_map = {
        "Billing": "Tier-1 / Billing & Accounts Receivable",
        "Technical Issue": "Tier-2 / Engineering Operations",
        "Account": "Tier-1 / Identity & Security Support",
        "General Query": "Tier-1 / Product Specialists"
    }
    
    return {
        "cat_pred": cat_pred,
        "cat_conf": cat_conf,
        "cat_probas": dict(zip(system['cat_pipe'].classes_, cat_probas)),
        "pri_pred": final_pri,
        "pri_conf": pri_conf,
        "pri_probas": dict(zip(system['pri_pipe'].classes_, pri_probas)),
        "sla_triggered": sla_triggered,
        "matched_rule": matched_rule,
        "human_flag": human_flag,
        "queue": queue_map.get(cat_pred, "General Queue"),
        "clean": clean
    }

# ==============================================================================
# TAB 1: REAL-TIME TRIAGE STUDIO
# ==============================================================================
with tab_triage:
    st.subheader("Interactive Support Ticket Triage Simulator")
    st.write("Click a quick preset scenario below to automatically populate the ticket and run triage:")

    # Quick Preset Buttons in 5 Columns
    p_cols = st.columns(5)
    for idx, (p_name, p_val) in enumerate(PRESETS.items()):
        if p_cols[idx].button(p_name, key=f"btn_p_{idx}"):
            st.session_state['input_subj'] = p_val['subj']
            st.session_state['input_body'] = p_val['body']
            st.session_state['run_now'] = True

    # Default session state initialization
    if 'input_subj' not in st.session_state:
        st.session_state['input_subj'] = "CRITICAL: Database connection timeout in production cluster"
    if 'input_body' not in st.session_state:
        st.session_state['input_body'] = "All API endpoints are returning HTTP 500 error since 10:00 AM UTC. Customer checkout is completely offline!"

    st.markdown("---")
    
    # Direct reactive text inputs (NO FORM BLOCKING)
    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        subj_text = st.text_input("Ticket Subject", value=st.session_state['input_subj'], key="text_subj")
    with col_t2:
        st.write("")
        st.caption("💡 Tip: Type or paste any support ticket text. Try typing keywords like '500 error', 'refund', '2FA', or 'dark mode'.")
        
    body_text = st.text_area("Ticket Body", value=st.session_state['input_body'], height=110, key="text_body")

    trigger_classify = st.button("🚀 Classify & Prioritize Ticket", type="primary", key="main_classify_btn")

    # Run prediction if user clicked button OR clicked a preset
    if trigger_classify or st.session_state.get('run_now', False):
        st.session_state['run_now'] = False
        
        with st.spinner("Analyzing semantics and checking SLA guardrails..."):
            res = execute_prediction(subj_text, body_text)
            st.session_state['last_res'] = res

    # Display results if available
    if 'last_res' in st.session_state:
        res = st.session_state['last_res']
        
        st.markdown("### 🎯 Decision-Support Output")
        
        # SLA Guardrail Banner
        if res['sla_triggered']:
            st.markdown(f"""
            <div class="sla-alert-box">
                <span style="font-size: 1.1rem; font-weight: 800;">🚨 CRITICAL SLA BREACH GUARDRAIL TRIGGERED</span><br/>
                Deterministic safety rule matched: <strong><code>{res['matched_rule']}</code></strong><br/>
                Priority automatically escalated to <strong>HIGH (SLA OVERRIDE)</strong>. Escalating to PagerDuty on-call queue.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sla-safe-box">
                <span style="font-weight: 700;">✅ Standard SLA Track:</span> No emergency safety violations detected. Urgency calculated via class-weighted ML model.
            </div>
            """, unsafe_allow_html=True)

        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.markdown("**Predicted Functional Queue**")
            st.markdown(f"<span class='badge-pill badge-cat'>{res['cat_pred']}</span>", unsafe_allow_html=True)
            st.metric("Category Confidence", f"{res['cat_conf']*100:.1f}%")
            st.caption(f"**Target Routing Squad:** {res['queue']}")
            
        with col_r2:
            st.markdown("**Assigned Severity Tier**")
            b_class = f"badge-{res['pri_pred'].lower()}"
            st.markdown(f"<span class='badge-pill {b_class}'>{res['pri_pred']} PRIORITY</span>", unsafe_allow_html=True)
            st.metric("Priority Confidence", f"{res['pri_conf']*100:.1f}%")
            if res['sla_triggered']:
                st.caption("🔒 Escalated via Deterministic Guardrail")
            else:
                st.caption("⚡ Assigned via Balanced ML Model")
                
        with col_r3:
            st.markdown("**Supervisory Action**")
            if res['human_flag']:
                st.warning("⚠️ Human Supervisor Review Required")
                st.caption(f"Threshold cutoff: {confidence_threshold*100:.0f}%")
            else:
                st.success("✅ Autonomous Direct Routing Approved")
                st.caption("Confidence exceeds threshold. Ticket dispatched.")

        # Class Probability Charts
        st.markdown("---")
        st.markdown("#### 🔬 Model Confidence Distribution")
        pcol1, pcol2 = st.columns(2)
        
        with pcol1:
            fig_cat = px.bar(
                x=list(res['cat_probas'].values()),
                y=list(res['cat_probas'].keys()),
                orientation='h',
                labels={'x': 'Probability', 'y': 'Category'},
                title="Category Class Probabilities",
                color=list(res['cat_probas'].values()),
                color_continuous_scale="Blues"
            )
            fig_cat.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_cat, key="plotly_cat")
            
        with pcol2:
            fig_pri = px.bar(
                x=list(res['pri_probas'].values()),
                y=list(res['pri_probas'].keys()),
                orientation='h',
                labels={'x': 'Probability', 'y': 'Priority'},
                title="Priority Class Probabilities",
                color=list(res['pri_probas'].values()),
                color_continuous_scale="YlOrRd"
            )
            fig_pri.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_pri, key="plotly_pri")

        with st.expander("🔍 View Preprocessed Clean Token Feature"):
            st.code(res['clean'])

# ==============================================================================
# TAB 2: BATCH CSV PROCESSING
# ==============================================================================
with tab_batch:
    st.subheader("Batch Support Ticket Processing")
    st.write("Run automated triage on multiple incoming support tickets simultaneously.")

    if st.button("🎲 Run Batch Triage on 50 Sampled Tickets", type="secondary", key="batch_run_btn"):
        sample_df = system['df'].sample(50, random_state=random.randint(1, 1000)).copy()
        
        batch_results = []
        for _, r in sample_df.iterrows():
            t_res = execute_prediction(r['ticket_subject'], r['ticket_body'])
            batch_results.append({
                "ticket_id": r['ticket_id'],
                "subject": r['ticket_subject'],
                "predicted_category": t_res['cat_pred'],
                "category_confidence": round(t_res['cat_conf'], 4),
                "predicted_priority": t_res['pri_pred'],
                "priority_confidence": round(t_res['pri_conf'], 4),
                "assigned_queue": t_res['queue'],
                "sla_override": t_res['sla_triggered'],
                "human_triage": t_res['human_flag']
            })
            
        res_df = pd.DataFrame(batch_results)
        st.session_state['batch_res'] = res_df

    if 'batch_res' in st.session_state:
        res_df = st.session_state['batch_res']
        st.success(f"Successfully processed {len(res_df)} tickets!")
        
        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        bcol1.metric("Total Batch Tickets", len(res_df))
        bcol2.metric("High Priority (P0/P1)", len(res_df[res_df['predicted_priority'] == 'High']))
        bcol3.metric("SLA Overrides Triggered", len(res_df[res_df['sla_override'] == True]))
        bcol4.metric("Human Audit Recommended", len(res_df[res_df['human_triage'] == True]))
        
        st.dataframe(res_df, use_container_width=True)
        
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Triaged Tickets CSV",
            data=csv_data,
            file_name="triaged_support_tickets.csv",
            mime="text/csv",
            key="dl_batch_btn"
        )

# ==============================================================================
# TAB 3: MODEL PERFORMANCE & CONFUSION MATRICES
# ==============================================================================
with tab_analytics:
    st.subheader("Model Validation & Confusion Matrices")
    st.write("Rigorous 5-Fold Stratified Cross-Validation metrics and test set confusion heatmaps.")

    mcol1, mcol2 = st.columns(2)
    
    with mcol1:
        st.markdown("#### Category Classifier Matrix (Multinomial Logistic Regression)")
        cm_c = system['cat_cm']
        labels_c = system['cat_classes']
        fig_cm_c = px.imshow(
            cm_c,
            x=labels_c,
            y=labels_c,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="Predicted Queue", y="True Queue", color="Count")
        )
        fig_cm_c.update_layout(height=400)
        st.plotly_chart(fig_cm_c, key="plotly_cm_cat")
        st.metric("Category Test Accuracy", f"{system['cat_acc']*100:.1f}%")
        
    with mcol2:
        st.markdown("#### Priority Predictor Matrix (Balanced Class Weights)")
        cm_p = system['pri_cm']
        labels_p = system['pri_classes']
        fig_cm_p = px.imshow(
            cm_p,
            x=labels_p,
            y=labels_p,
            color_continuous_scale="YlOrRd",
            text_auto=True,
            labels=dict(x="Predicted Priority", y="True Priority", color="Count")
        )
        fig_cm_p.update_layout(height=400)
        st.plotly_chart(fig_cm_p, key="plotly_cm_pri")
        st.metric("Priority Test Accuracy", f"{system['pri_acc']*100:.1f}%")

# ==============================================================================
# TAB 4: LINGUISTIC EXPLAINABILITY
# ==============================================================================
with tab_explain:
    st.subheader("Top Predictive TF-IDF N-Grams (Feature Attribution)")
    st.write("Inspect the exact mathematical log-odds coefficients (β weights) driving classification decisions.")

    subtab_cat, subtab_pri = st.tabs(["Category Linguistic Signals", "Priority Linguistic Signals"])
    
    with subtab_cat:
        selected_cat = st.selectbox("Select Functional Queue to Inspect:", system['cat_classes'], key="sb_explain_cat")
        kw_data = system['cat_top_kw'][selected_cat]
        df_kw = pd.DataFrame(kw_data, columns=['N-Gram', 'Weight']).sort_values('Weight', ascending=True)
        
        fig_kw = px.bar(
            df_kw,
            x='Weight',
            y='N-Gram',
            orientation='h',
            title=f"Top 10 Predictive Terms for '{selected_cat}'",
            color='Weight',
            color_continuous_scale="Viridis"
        )
        fig_kw.update_layout(height=400)
        st.plotly_chart(fig_kw, key="plotly_kw_cat")
        
    with subtab_pri:
        selected_pri = st.selectbox("Select Urgency Tier to Inspect:", system['pri_classes'], key="sb_explain_pri")
        kw_p_data = system['pri_top_kw'][selected_pri]
        df_p_kw = pd.DataFrame(kw_p_data, columns=['N-Gram', 'Weight']).sort_values('Weight', ascending=True)
        
        fig_p_kw = px.bar(
            df_p_kw,
            x='Weight',
            y='N-Gram',
            orientation='h',
            title=f"Top 10 Urgency Terms for '{selected_pri} Priority'",
            color='Weight',
            color_continuous_scale="Plasma"
        )
        fig_p_kw.update_layout(height=400)
        st.plotly_chart(fig_p_kw, key="plotly_kw_pri")

# ==============================================================================
# TAB 5: EXECUTIVE ROI & SLA IMPACT CALCULATOR
# ==============================================================================
with tab_roi:
    st.subheader("SaaS Executive ROI & Support Capacity Simulator")
    st.write("Adjust operational parameters below to model the annual labor savings and SLA breach prevention value.")

    rcol1, rcol2 = st.columns([1, 1])
    
    with rcol1:
        monthly_tickets = st.slider("Monthly Ticket Influx", 1000, 50000, 15000, 1000, key="sl_vol")
        triage_sec_manual = st.slider("Manual Triage Time per Ticket (Seconds)", 15, 120, 45, 5, key="sl_sec")
        agent_hourly_rate = st.slider("Blended Agent Hourly Cost ($/hr)", 20, 75, 35, 5, key="sl_wage")
        sla_breach_cost = st.slider("Average Cost of SLA Penalty / Breach ($)", 500, 10000, 2500, 500, key="sl_pen")
        
    with rcol2:
        hours_saved_month = (monthly_tickets * (triage_sec_manual / 3600)) * 0.95
        annual_labor_savings = hours_saved_month * 12 * agent_hourly_rate
        breaches_prevented_yr = int((monthly_tickets * 12) * 0.005)
        annual_sla_penalty_savings = breaches_prevented_yr * sla_breach_cost
        total_annual_economic_value = annual_labor_savings + annual_sla_penalty_savings
        
        st.markdown("### 💰 Projected Annual ROI")
        st.metric("Total Annual Economic Value", f"${total_annual_economic_value:,.0f}")
        st.metric("Annual Direct Labor Reclaimed", f"${annual_labor_savings:,.0f}", f"{hours_saved_month*12:,.0f} hours/yr")
        st.metric("SLA Breach Penalties Averted", f"${annual_sla_penalty_savings:,.0f}", f"{breaches_prevented_yr} incidents/yr")
