"""
src/models.py
=============
Pipelines for Category Classification and Imbalanced Priority Prediction.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


def build_tfidf_vectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True, min_df=2):
    """Returns a production TF-IDF vectorizer configuration."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=min_df
    )


def create_category_pipeline(model_type='logistic_regression', random_state=42):
    """
    Creates a category classification pipeline with TF-IDF and specified classifier.
    Supported: 'logistic_regression', 'linear_svc', 'random_forest'
    """
    tfidf = build_tfidf_vectorizer()
    if model_type == 'logistic_regression':
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
    elif model_type == 'linear_svc':
        clf = CalibratedClassifierCV(LinearSVC(C=1.0, random_state=random_state, dual=False), cv=3)
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
        
    return Pipeline([('tfidf', tfidf), ('clf', clf)])


def create_priority_pipeline(random_state=42):
    """
    Creates a class-weighted priority classification pipeline to handle urgency imbalance.
    """
    tfidf = build_tfidf_vectorizer()
    clf = LogisticRegression(class_weight='balanced', C=1.2, max_iter=1000, random_state=random_state)
    return Pipeline([('tfidf', tfidf), ('clf', clf)])


def extract_top_features_per_class(pipeline, top_k=10):
    """
    Extracts the top-weighted TF-IDF terms per target class from model coefficients.
    """
    vec = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    
    if hasattr(clf, 'calibrated_classifiers_'):
        base_clf = clf.calibrated_classifiers_[0].estimator
        coefs = base_clf.coef_
        classes = base_clf.classes_
    else:
        coefs = clf.coef_
        classes = clf.classes_
        
    feature_names = np.array(vec.get_feature_names_out())
    results = {}
    
    for idx, class_name in enumerate(classes):
        top_indices = np.argsort(coefs[idx])[-top_k:][::-1]
        results[class_name] = list(zip(feature_names[top_indices], coefs[idx][top_indices]))
        
    return results
