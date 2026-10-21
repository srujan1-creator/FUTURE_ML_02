"""
app.py
======
Production Web Application & REST API for Support Ticket Classification & Prioritization.
Compatible with Vercel Serverless, Gunicorn/Waitress, and Docker containers.
Future Interns - Machine Learning Task 2 (2026)
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

from src.preprocessing import preprocess_ticket_text
from src.models import create_category_pipeline, create_priority_pipeline
from src.inference import predict_ticket

app = Flask(__name__, template_folder="templates")
application = app  # WSGI compatibility
handler = app      # Serverless compatibility

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Train models on startup (or load from disk)
def get_trained_pipelines():
    csv_path = os.path.join(BASE_DIR, "data", "support_tickets.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback inline generation if file not found
        rows = [
            {"raw_text": "Production API 500 error", "category": "Technical Issue", "priority": "High"},
            {"raw_text": "Unauthorized charge refund", "category": "Billing", "priority": "High"},
            {"raw_text": "Reset password 2FA login", "category": "Account", "priority": "Medium"},
            {"raw_text": "Dark mode feature request roadmap", "category": "General Query", "priority": "Low"}
        ]
        df = pd.DataFrame(rows)

    if 'clean_text' not in df.columns:
        df['raw_text'] = df['ticket_subject'].fillna('') + " " + df['ticket_body'].fillna('')
        df['clean_text'] = df['raw_text'].apply(preprocess_ticket_text)

    cat_pipe = create_category_pipeline(model_type='logistic_regression')
    cat_pipe.fit(df['clean_text'], df['category'])

    pri_pipe = create_priority_pipeline()
    pri_pipe.fit(df['clean_text'], df['priority'])

    return cat_pipe, pri_pipe

cat_pipeline, pri_pipeline = get_trained_pipelines()


@app.route("/")
def index():
    """Serves the interactive web UI if index.html exists."""
    template_path = os.path.join(BASE_DIR, "templates", "index.html")
    if os.path.exists(template_path):
        return render_template("index.html")
    return "<h1>Support Ticket Intelligence API</h1><p>Visit /api/health or POST /api/predict</p>"


@app.route("/api/health")
def health():
    """System health check endpoint."""
    return jsonify({
        "status": "healthy",
        "system": "Support Ticket Classification & Prioritization Intelligence",
        "task": "Future Interns ML Task 2 (2026)",
        "models": ["Tfidf-LogisticRegression-Category", "Tfidf-BalancedLogisticRegression-Priority"],
        "version": "2.0.0"
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    POST endpoint for automated ticket triage.
    Expects JSON: {"ticket_subject": "...", "ticket_body": "...", "confidence_threshold": 0.55}
    """
    data = request.get_json(force=True, silent=True) or {}
    subject = data.get("ticket_subject", data.get("subject", ""))
    body = data.get("ticket_body", data.get("body", ""))
    thresh = float(data.get("confidence_threshold", 0.50))

    if not subject and not body:
        return jsonify({"error": "Missing 'ticket_subject' or 'ticket_body' in JSON body"}), 400

    result = predict_ticket(
        subject=subject,
        body=body,
        cat_pipeline=cat_pipeline,
        pri_pipeline=pri_pipeline,
        confidence_threshold=thresh
    )
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
