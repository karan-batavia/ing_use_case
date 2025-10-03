#!/usr/bin/env python3
"""
Streamlit App for Sensitivity Classifier
- Test individual texts for sensitivity classification
- Visualize model performance metrics
- Display confusion matrix and classification metrics
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import re
import subprocess
import sys
import os
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Any, Optional

# Import or define the same patterns and RuleFeatureizer from ml_setup.py
FALLBACK_PATTERNS: Dict[str, str] = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE_EU": r"(?:\+\d{1,3}\s?)?(?:\d[\s-]?){9,}",
    "SSN_LIKE": r"\b\d{6}[- ]?\d{2,4}[\.]?\d{0,2}\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    "ACCOUNT_NUM": r"\b(?:acct|account)\s*\d{3,}\b",
    "AMOUNT": r"(?:USD|EUR|GBP|€|\$|£)\s?\d{1,3}(?:[, \u00A0]\d{3})*(?:\.\d{2})?",
    "DOB": r"\b\d{4}-\d{2}-\d{2}\b",
    "NATIONAL_ID": r"\bID[:\s-]?[A-Z0-9]{6,}\b",
    "BIOMETRIC": r"\b(FaceID|fingerprint|iris|biometric)\b",
}

PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in FALLBACK_PATTERNS.items()}

class RuleFeatureizer(BaseEstimator, TransformerMixin):
    """Rule-based feature extractor - must match ml_setup.py exactly"""
    def __init__(self, pattern_keys: Optional[List[str]] = None):
        self.pattern_keys = pattern_keys or list(PATTERNS.keys())

    def fit(self, X: List[str], y=None):
        return self

    def transform(self, X: List[str]) -> Any:
        feats = []
        for text in X:
            row = []
            for k in self.pattern_keys:
                p = PATTERNS[k]
                matches = p.findall(text)
                row.append(int(bool(matches)))  # presence
                row.append(len(matches))        # count
            def has_any(words):
                tl = text.lower()
                return int(any(w in tl for w in words))
            row.extend([
                has_any(["credit score","income","account balance","masked pin","biometric"]),
                has_any(["agreement","supplier","customer","standing order","payment order","overdraft"]),
                has_any(["annual report","pillar 3","press","newsroom","full year results","investor"]),
                has_any(["policy","guideline","standard","sop","governance","raci","deprecated","retired"]),
            ])
            feats.append(row)
        return np.array(feats, dtype=float)

# Page config
st.set_page_config(
    page_title="Sensitivity Classifier",
    page_icon="🔒",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = Path("sensitivity_classifier.joblib")
    if not model_path.exists():
        return None
    try:
        data = joblib.load(model_path)
        return data['pipeline'], data['labels']
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_text(pipeline, text):
    """Predict sensitivity level for a single text"""
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    return prediction, probabilities

def redact_sensitive_info(text):
    """Replace sensitive information with placeholders"""
    redacted_text = text
    detections = []
    
    # Define placeholders for each pattern type
    placeholders = {
        "EMAIL": "[EMAIL]",
        "PHONE_EU": "[PHONE]",
        "SSN_LIKE": "[SSN]",
        "IBAN": "[IBAN]",
        "ACCOUNT_NUM": "[ACCOUNT_NUMBER]",
        "AMOUNT": "[AMOUNT]",
        "DOB": "[DATE_OF_BIRTH]",
        "NATIONAL_ID": "[NATIONAL_ID]",
        "BIOMETRIC": "[BIOMETRIC_DATA]",
    }
    
    # Track all matches with their positions (in reverse order to maintain indices)
    matches = []
    for pattern_name, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            matches.append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group(),
                'type': pattern_name,
                'placeholder': placeholders.get(pattern_name, f"[{pattern_name}]")
            })
    
    # Sort by position (reverse order to maintain string indices during replacement)
    matches.sort(key=lambda x: x['start'], reverse=True)
    
    # Replace matches with placeholders
    for match in matches:
        redacted_text = (
            redacted_text[:match['start']] + 
            match['placeholder'] + 
            redacted_text[match['end']:]
        )
        detections.append({
            'type': match['type'],
            'original': match['text'],
            'placeholder': match['placeholder']
        })
    
    # Reverse detections to show in original order
    detections.reverse()
    
    return redacted_text, detections

def create_confusion_matrix_plot(cm_data, labels):
    """Create an interactive confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm_data,
        texttemplate='%{text}',
        textfont={"size": 16},
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=500,
        xaxis={'side': 'bottom'}
    )
    
    return fig

def create_metrics_bar_chart(metrics_df):
    """Create bar chart for precision, recall, f1-score"""
    fig = go.Figure()
    
    for metric in ['precision', 'recall', 'f1-score']:
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=metrics_df.index,
            y=metrics_df[metric],
            text=metrics_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Classification Metrics by Category",
        xaxis_title="Category",
        yaxis_title="Score",
        barmode='group',
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def create_probability_chart(probabilities, labels):
    """Create bar chart for prediction probabilities"""
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities,
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Category",
        yaxis_title="Probability",
        height=300,
        yaxis=dict(range=[0, 1], tickformat='.0%')
    )
    
    return fig

def train_model(data_path, model_choice='logreg'):
    """Train the model using ml_setup.py"""
    try:
        # Set environment variables
        env = os.environ.copy()
        env['MODEL'] = model_choice
        env['DATA_PATH'] = data_path
        
        # Run training script
        result = subprocess.run(
            [sys.executable, 'ml_setup.py', '--train'],
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            # Parse output to extract metrics
            output = result.stdout
            
            # Extract classification report and confusion matrix from output
            lines = output.split('\n')
            report_start = None
            cm_start = None
            
            for i, line in enumerate(lines):
                if 'Classification Report' in line:
                    report_start = i + 1
                elif 'Confusion Matrix' in line:
                    cm_start = i + 1
            
            # Extract report and confusion matrix
            report = ""
            confusion_matrix = None
            
            if report_start:
                for i in range(report_start, len(lines)):
                    if 'Confusion Matrix' in lines[i]:
                        break
                    report += lines[i] + '\n'
            
            if cm_start and cm_start < len(lines):
                try:
                    cm_line = lines[cm_start].strip()
                    confusion_matrix = json.loads(cm_line)
                except:
                    pass
            
            return {
                'success': True,
                'output': output,
                'report': report.strip(),
                'confusion_matrix': confusion_matrix
            }
        else:
            return {
                'success': False,
                'error': result.stderr or result.stdout
            }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'Training timeout exceeded (5 minutes)'
        }
    except FileNotFoundError:
        return {
            'success': False,
            'error': 'ml_setup.py not found in current directory'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def parse_classification_report(report_text):
    """Parse sklearn classification report into DataFrame"""
    lines = report_text.strip().split('\n')
    data = []
    summary_data = {}
    
    # Parse main categories
    for line in lines:
        parts = line.split()
        if len(parts) >= 5 and parts[0] in ['C1', 'C2', 'C3', 'C4']:
            data.append({
                'Category': parts[0],
                'precision': float(parts[1]),
                'recall': float(parts[2]),
                'f1-score': float(parts[3]),
                'support': int(parts[4])
            })
        # Parse summary lines
        elif 'accuracy' in line.lower():
            parts = line.split()
            summary_data['accuracy'] = float(parts[1])
            summary_data['accuracy_support'] = int(parts[2])
        elif 'macro avg' in line.lower():
            summary_data['macro_precision'] = float(parts[2])
            summary_data['macro_recall'] = float(parts[3])
            summary_data['macro_f1'] = float(parts[4])
        elif 'weighted avg' in line.lower():
            summary_data['weighted_precision'] = float(parts[2])
            summary_data['weighted_recall'] = float(parts[3])
            summary_data['weighted_f1'] = float(parts[4])
    
    df = pd.DataFrame(data).set_index('Category')
    return df, summary_data

# Main app
def main():
    st.title("🔒 Sensitivity Classifier Dashboard")
    st.markdown("Test and evaluate the hybrid sensitivity classification model")
    
    # Load model
    model_data = load_model()
    
    if model_data is None:
        st.error("⚠️ Model not found! Please train the model first using `ml_setup.py --train`")
        st.info("Run: `python ml_setup.py --train` to train and save the model")
        return
    
    pipeline, labels = model_data
    
    # Sidebar
    st.sidebar.header("Model Information")
    st.sidebar.info(f"""
    **Labels:** {', '.join(labels)}
    
    **Categories:**
    - **C1**: Public (reports, press)
    - **C2**: Internal (policies, SOPs)
    - **C3**: Confidential (transactions)
    - **C4**: Highly Sensitive (PII, financial)
    """)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Test Classifier", "🔒 Redact Sensitive Data", "🎯 Train Model", "📊 Model Performance"])
    
    # Tab 1: Test Classifier
    with tab1:
        st.header("Test Your Text")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            test_text = st.text_area(
                "Enter text to classify:",
                height=150,
                placeholder="E.g., 'Customer agreement for standing order EUR 1,500 monthly payment'"
            )
            
            # Example texts
            st.markdown("**Quick Examples:**")
            examples = {
                "Public Report": "Annual report 2024 full year results for investors and analysts",
                "Internal Policy": "Updated SOP for review cycle and governance approval process",
                "Transaction": "Wire transfer EUR 5,000 to supplier account for invoice PO-12345",
                "Sensitive PII": "Customer email john.doe@example.com national ID BE123456789 account balance EUR 15,000"
            }
            
            example_cols = st.columns(4)
            for i, (name, text) in enumerate(examples.items()):
                if example_cols[i].button(name, use_container_width=True):
                    test_text = text
                    st.rerun()
        
        with col2:
            if st.button("🔍 Classify", type="primary", use_container_width=True):
                if test_text.strip():
                    with st.spinner("Analyzing..."):
                        prediction, probabilities = predict_text(pipeline, test_text)
                        
                        # Display prediction
                        st.success(f"**Predicted Category: {prediction}**")
                        
                        # Show probability distribution
                        st.plotly_chart(
                            create_probability_chart(probabilities, labels),
                            use_container_width=True
                        )
                        
                        # Category explanation
                        explanations = {
                            'C1': '📄 Public information suitable for external sharing',
                            'C2': '🏢 Internal use only - policies and procedures',
                            'C3': '🔐 Confidential - business transactions',
                            'C4': '⚠️ Highly Sensitive - contains PII or financial data'
                        }
                        st.info(explanations.get(prediction, ''))
                else:
                    st.warning("Please enter some text to classify")
    
    # Tab 2: Redact Sensitive Data
    with tab2:
        st.header("🔒 Redact Sensitive Information")
        st.markdown("Automatically detect and replace **all** sensitive information with placeholders")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📝 Original Text")
            redact_text = st.text_area(
                "Enter text with sensitive information:",
                height=250,
                placeholder="E.g., 'Please contact john.doe@example.com or call +32 475 12 34 56 regarding account BE68 5390 0754 7034 for EUR 1,500 payment'"
            )
            
            redact_button = st.button("🔒 Redact All Sensitive Data", type="primary", use_container_width=True)
            
            st.info("💡 All detected sensitive information will be automatically redacted")
        
        with col2:
            st.subheader("✅ Redacted Text")
            
            if redact_button and redact_text.strip():
                redacted_text, detections = redact_sensitive_info(redact_text)
                
                # Display redacted text
                st.text_area(
                    "Redacted version:",
                    value=redacted_text,
                    height=250,
                    disabled=True
                )
                
                # Show statistics
                if detections:
                    total_redacted = len(detections)
                    st.success(f"✅ Successfully redacted **{total_redacted}** sensitive item(s)")
                else:
                    st.success("✨ No sensitive information detected - text is clean!")
                
                # Copy button
                st.download_button(
                    label="📥 Download Redacted Text",
                    data=redacted_text,
                    file_name="redacted_text.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                # Show what was detected and redacted
                if detections:
                    st.markdown("---")
                    st.subheader("🔍 Detected Sensitive Information")
                    
                    # Create detection summary
                    detection_df = pd.DataFrame(detections)
                    detection_counts = detection_df['type'].value_counts()
                    
                    # Show counts
                    st.markdown("**Detection Summary:**")
                    count_cols = st.columns(min(4, len(detection_counts)))
                    for idx, (det_type, count) in enumerate(detection_counts.items()):
                        count_cols[idx % 4].metric(det_type, count)
                    
                    # Show detailed table
                    st.markdown("**Redaction Details:**")
                    display_detections = detection_df.copy()
                    display_detections.columns = ['Type', 'Original Value', 'Replaced With']
                    
                    st.dataframe(
                        display_detections.style.set_properties(**{
                            'text-align': 'left',
                            'font-size': '13px'
                        }),
                        use_container_width=True,
                        height=min(400, len(detections) * 35 + 50)
                    )
                    
                    # Visualization of detection types
                    if len(detection_counts) > 1:
                        fig = px.pie(
                            values=detection_counts.values,
                            names=detection_counts.index,
                            title="Distribution of Detected Sensitive Data Types",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("✨ No sensitive information detected in the text")
            
            elif redact_button:
                st.warning("Please enter some text to redact")
        
        # Information section
        st.markdown("---")
        st.markdown("### 📚 Supported Sensitive Data Types")
        
        info_cols = st.columns(3)
        
        with info_cols[0]:
            st.markdown("""
            **Personal Identifiers:**
            - 📧 Email addresses
            - 📞 Phone numbers
            - 🆔 National ID numbers
            - 🔢 SSN-like numbers
            """)
        
        with info_cols[1]:
            st.markdown("""
            **Financial Information:**
            - 🏦 IBAN numbers
            - 💳 Account numbers
            - 💰 Monetary amounts
            - 📅 Dates of birth
            """)
        
        with info_cols[2]:
            st.markdown("""
            **Biometric Data:**
            - 👤 FaceID references
            - 👆 Fingerprint data
            - 👁️ Iris scan data
            - 🔐 Biometric identifiers
            """)
    
    # Tab 3: Train Model
    with tab3:
        st.header("🎯 Train Sensitivity Classifier")
        
        st.markdown("""
        Train a new model or retrain with different parameters. The training process will:
        1. Load your dataset
        2. Apply hybrid rules + ML classification
        3. Generate performance metrics
        4. Save the trained model
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Data file selection
            data_file = st.text_input(
                "Training Data File",
                value="raw_prompts.txt",
                help="Path to your training data file (one prompt per line)"
            )
            
            # Check if file exists
            if Path(data_file).exists():
                st.success(f"✓ File found: {data_file}")
                with open(data_file, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]
                    st.info(f"📄 Dataset contains {len(lines)} samples")
            else:
                st.error(f"✗ File not found: {data_file}")
        
        with col2:
            # Model selection
            model_type = st.selectbox(
                "Model Type",
                options=["logreg", "rf"],
                help="logreg: Logistic Regression (fast, reliable)\nrf: Random Forest (more complex)"
            )
            
            st.markdown("**Model Details:**")
            if model_type == "logreg":
                st.info("🔹 Logistic Regression\n- Fast training\n- Good interpretability\n- Recommended for most cases")
            else:
                st.info("🌲 Random Forest\n- Slower training\n- Can capture complex patterns\n- 300 estimators")
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                test_size = st.slider("Test Split %", 10, 40, 20) / 100
            with col_b:
                ngram_max = st.slider("Max N-gram", 1, 3, 2)
            with col_c:
                random_seed = st.number_input("Random Seed", value=42)
        
        # Train button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not Path(data_file).exists():
                st.error("❌ Cannot start training: Data file not found!")
            else:
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("⏳ Initializing training...")
                progress_bar.progress(10)
                
                # Set environment variables for training
                env = os.environ.copy()
                env['MODEL'] = model_type
                env['DATA_PATH'] = data_file
                env['TEST_SIZE'] = str(test_size)
                env['NGRAM_MAX'] = str(ngram_max)
                env['SEED'] = str(random_seed)
                
                status_text.text("🔄 Training model...")
                progress_bar.progress(30)
                
                # Run training
                result = train_model(data_file, model_type)
                
                progress_bar.progress(90)
                
                if result['success']:
                    progress_bar.progress(100)
                    status_text.text("✅ Training completed!")
                    
                    st.success("🎉 Model trained successfully!")
                    
                    # Display results
                    st.markdown("### 📊 Training Results")
                    
                    # Parse and display metrics
                    if result['report']:
                        try:
                            df_metrics, summary = parse_classification_report(result['report'])
                            
                            # Display key metrics
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            metric_col1.metric("Accuracy", f"{summary.get('accuracy', 0):.1%}")
                            metric_col2.metric("Macro F1", f"{summary.get('macro_f1', 0):.3f}")
                            metric_col3.metric("Weighted F1", f"{summary.get('weighted_f1', 0):.3f}")
                            metric_col4.metric("Samples", summary.get('accuracy_support', 0))
                            
                            # Display metrics table
                            st.markdown("#### Per-Category Performance")
                            display_df = df_metrics.copy()
                            display_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
                            
                            st.dataframe(
                                display_df.style.format({
                                    'Precision': '{:.3f}',
                                    'Recall': '{:.3f}',
                                    'F1-Score': '{:.3f}',
                                    'Support': '{:.0f}'
                                }).background_gradient(
                                    subset=['Precision', 'Recall', 'F1-Score'],
                                    cmap='RdYlGn',
                                    vmin=0,
                                    vmax=1
                                ),
                                use_container_width=True
                            )
                            
                            # Metrics chart
                            st.plotly_chart(
                                create_metrics_bar_chart(df_metrics),
                                use_container_width=True
                            )
                            
                            # Confusion matrix
                            if result['confusion_matrix']:
                                st.markdown("#### Confusion Matrix")
                                cm_array = np.array(result['confusion_matrix'])
                                fig_cm = create_confusion_matrix_plot(cm_array, labels)
                                st.plotly_chart(fig_cm, use_container_width=True)
                            
                            # Save metrics to file
                            metrics_data = {
                                'report': result['report'],
                                'confusion_matrix': result['confusion_matrix'],
                                'model_type': model_type,
                                'data_file': data_file,
                                'test_size': test_size,
                                'accuracy': summary.get('accuracy', 0)
                            }
                            with open('model_metrics.json', 'w') as f:
                                json.dump(metrics_data, f, indent=2)
                            
                            st.info("💾 Metrics saved to `model_metrics.json`")
                            
                        except Exception as e:
                            st.warning(f"Could not parse metrics: {e}")
                            with st.expander("📄 View Raw Output"):
                                st.code(result['output'])
                    
                    # Show full output
                    with st.expander("📄 View Full Training Log"):
                        st.code(result['output'])
                    
                    st.info("🔄 Switch to the 'Model Performance' tab to see detailed analysis or reload the page to test the new model.")
                    
                else:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                    with st.expander("🔍 View Error Details"):
                        st.code(result.get('error', 'No details available'))
    
    # Tab 4: Model Performance
    with tab4:
        st.header("Model Performance Metrics")
        
        # Load metrics if available
        metrics_file = Path("model_metrics.json")
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Display overall accuracy
            col1, col2, col3 = st.columns(3)
            
            try:
                df_metrics, summary = parse_classification_report(metrics_data['report'])
                col1.metric("Overall Accuracy", f"{summary.get('accuracy', 0):.3f}")
                col2.metric("Macro Avg F1-Score", f"{summary.get('macro_f1', 0):.3f}")
                col3.metric("Total Samples", summary.get('accuracy_support', 0))
            except Exception as e:
                st.warning(f"Could not parse summary metrics: {e}")
            
            st.markdown("---")
            
            # Metrics table
            st.subheader("📈 Classification Metrics by Category")
            
            try:
                df_metrics, summary = parse_classification_report(metrics_data['report'])
                
                # Create a clean display dataframe
                display_df = df_metrics.copy()
                display_df.columns = ['Precision', 'Recall', 'F1-Score', 'Support']
                
                # Add summary rows
                summary_rows = pd.DataFrame([
                    {
                        'Precision': summary.get('macro_precision', 0),
                        'Recall': summary.get('macro_recall', 0),
                        'F1-Score': summary.get('macro_f1', 0),
                        'Support': summary.get('accuracy_support', 0)
                    },
                    {
                        'Precision': summary.get('weighted_precision', 0),
                        'Recall': summary.get('weighted_recall', 0),
                        'F1-Score': summary.get('weighted_f1', 0),
                        'Support': summary.get('accuracy_support', 0)
                    }
                ], index=['macro avg', 'weighted avg'])
                
                # Combine main metrics with summary
                full_df = pd.concat([display_df, summary_rows])
                
                # Format and display
                st.dataframe(
                    full_df.style.format({
                        'Precision': '{:.3f}',
                        'Recall': '{:.3f}',
                        'F1-Score': '{:.3f}',
                        'Support': '{:.0f}'
                    }).background_gradient(
                        subset=pd.IndexSlice[['C1', 'C2', 'C3', 'C4'], ['Precision', 'Recall', 'F1-Score']],
                        cmap='RdYlGn',
                        vmin=0,
                        vmax=1
                    ).set_properties(**{
                        'text-align': 'center',
                        'font-size': '14px'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center'), ('font-weight', 'bold')]},
                        {'selector': 'td', 'props': [('text-align', 'center')]},
                    ]),
                    use_container_width=True,
                    height=350
                )
                
                st.info(f"**Accuracy: {summary.get('accuracy', 0):.3f}** (correctly classified: {int(summary.get('accuracy', 0) * summary.get('accuracy_support', 0))}/{summary.get('accuracy_support', 0)} samples)")
                
                # Metrics bar chart - only for main categories
                st.plotly_chart(
                    create_metrics_bar_chart(df_metrics),
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Could not parse classification metrics: {e}")
            
            st.markdown("---")
            
            # Confusion Matrix
            st.subheader("🎯 Confusion Matrix")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                cm_array = np.array(metrics_data['confusion_matrix'])
                fig_cm = create_confusion_matrix_plot(cm_array, labels)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                st.markdown("**Understanding the Matrix:**")
                st.markdown("""
                - **Diagonal**: Correct predictions
                - **Off-diagonal**: Misclassifications
                - **Rows**: True labels
                - **Columns**: Predicted labels
                """)
                
                # Calculate accuracy per class
                cm_array = np.array(metrics_data['confusion_matrix'])
                for i, label in enumerate(labels):
                    if cm_array[i].sum() > 0:
                        acc = cm_array[i][i] / cm_array[i].sum()
                        st.metric(f"{label} Accuracy", f"{acc:.1%}")
            
            # Raw classification report
            with st.expander("📄 View Full Classification Report"):
                st.code(metrics_data['report'])
        
        else:
            st.warning("⚠️ No metrics file found. Metrics are saved during training.")
            st.info("""
            To generate metrics:
            1. Run `python ml_setup.py --train`
            2. The script will create `model_metrics.json`
            3. Refresh this page to see the metrics
            """)
            
            st.markdown("---")
            st.subheader("💡 Model Training Tips")
            st.markdown("""
            - **MODEL**: Choose `logreg` (default) or `rf` (random forest)
            - **MIN_DF**: Minimum document frequency for terms (default: 1)
            - **NGRAM_MAX**: Maximum n-gram size (default: 2)
            - **TEST_SIZE**: Test split ratio (default: 0.2)
            
            Example:
            ```bash
            MODEL=rf NGRAM_MAX=3 python ml_setup.py --train
            ```
            """)

if __name__ == "__main__":
    main()