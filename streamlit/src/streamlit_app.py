import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="Fraud Detection Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"


# ============================================================================
# Custom CSS for Premium Design
# ============================================================================

st.markdown("""
<style>
    /* Main background and text */
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Headers */
    h1 {
        color: #fff;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: #e0e0e0;
        font-weight: 600;
    }
    
    /* Cards */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Divider */
    hr {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 0;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }
    /* Custom styling for the Predictor result */
    .stAlert div[role="alert"] {
        padding: 1rem;
        border-radius: 10px;
    }
    .stAlert.error {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #ff4b4b;
    }
    .stAlert.success {
        background-color: rgba(0, 204, 136, 0.1);
        border-left: 5px solid #00cc88;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions (API Calls)
# ============================================================================

@st.cache_data(ttl=60)
def fetch_stats():
    """Fetch overall statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_transactions(limit=100, class_filter=None):
    """Fetch transactions"""
    try:
        params = {"limit": limit}
        if class_filter is not None:
            params["class"] = class_filter
        response = requests.get(f"{API_BASE_URL}/api/transactions", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching transactions: {e}")
        return []

@st.cache_data(ttl=60)
def fetch_amount_distribution(limit=1000):
    """Fetch amount distribution"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/distributions/amount",
            params={"limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching amount distribution: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_feature_distribution(feature_name, limit=1000):
    """Fetch distribution for any feature"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/distributions/feature/{feature_name}",
            params={"limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching feature distribution: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_scatter_data(x_feature, y_feature, limit=1000):
    """Fetch scatter plot data"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/scatter",
            params={"x_feature": x_feature, "y_feature": y_feature, "limit": limit},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching scatter data: {e}")
        return None

@st.cache_data(ttl=60)
def fetch_explanation(payload):
    """Fetch SHAP explanation for a transaction"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/explain", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching explanation: {e}")
        return None


# ============================================================================
# Reusable UI Components
# ============================================================================

def display_metrics(stats):
    """Display the 4 key metrics"""
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Volume", f"{stats['total_transactions']:,}")
        col2.metric("Detected Threats", f"{stats['fraud_count']:,}", f"{stats['fraud_rate']:.3f}%", delta_color="inverse")
        col3.metric("Verified Legitimate", f"{stats['normal_count']:,}")
        col4.metric("Current Fraud Rate", f"{stats['fraud_rate']:.3f}%", delta_color="inverse")
    else:
        st.warning("Unable to fetch statistics. Please ensure the API server is running.")

def display_recent_transactions():
    """Display recent transactions table with filters"""
    st.subheader("Recent Transactions")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        transaction_limit = st.slider("Rows to fetch", 10, 200, 50, 10, key="txn_limit_slider")
    with col2:
        class_filter_option = st.selectbox("Filter Scope", ["All", "Fraud Only", "Normal Only"], key="txn_filter_select")
    
    class_filter = "1" if class_filter_option == "Fraud Only" else "0" if class_filter_option == "Normal Only" else None
    
    transactions = fetch_transactions(limit=transaction_limit, class_filter=class_filter)
    
    if transactions:
        df = pd.DataFrame(transactions)
        df['Class'] = df['Class'].map({'0': 'Normal', '1': 'Fraud'})
        df['Amount'] = df['Amount'].apply(lambda x: f"${x:.2f}")
        
        display_cols = ['id', 'Amount', 'Class'] + [col for col in df.columns if col.startswith('V')][:5]
        
        st.dataframe(df[display_cols], width="stretch", height=400, hide_index=True)
        
        csv = df.to_csv(index=False)
        st.download_button("Export Data (CSV)", data=csv, file_name="fraud_transactions.csv", mime="text/csv", key="export_btn")
    else:
        st.info("No transactions available matching criteria")



# ============================================================================
# Page Views
# ============================================================================

def render_home():
    """Page 1: Dashboard (Executive Overview & Predictor)"""
    st.title("Dashboard") 
    
    # ------------------------------------------------------------------------
    # EXECUTIVE OVERVIEW (STATS)
    # ------------------------------------------------------------------------
    st.subheader("Transaction Anomaly Monitoring & Analysis")
    
    stats = fetch_stats()
    display_metrics(stats)
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # REALTIME FRAUD PREDICTOR SIMULATOR
    # ------------------------------------------------------------------------
    st.header("Live Transaction Monitor") 
    
    # Initialize session state for realtime simulator
    if 'simulator_running' not in st.session_state:
        st.session_state.simulator_running = False
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'transactions_processed' not in st.session_state:
        st.session_state.transactions_processed = 0
    if 'frauds_detected' not in st.session_state:
        st.session_state.frauds_detected = 0
    if 'latest_amount' not in st.session_state:
        st.session_state.latest_amount = 0.0
    
    # Load test data once
    try:
        if 'test_data' not in st.session_state:
            st.session_state.test_data = pd.read_csv("src/test.csv")
        df = st.session_state.test_data
    except FileNotFoundError:
        st.error("Could not find src/test.csv. Ensure test data file is present.")
        return
    
    # Control Panel & Metrics Layout
    col_controls, col_spacer, col_stat1, col_stat2, col_stat3 = st.columns([1, 0.5, 1, 1, 1])
    
    with col_controls:
        # Stack buttons vertically
        if st.button("START" if not st.session_state.simulator_running else "PAUSE", 
                    type="primary", use_container_width=True, key="toggle_sim"):
            st.session_state.simulator_running = not st.session_state.simulator_running
            
        if st.button("RESET", use_container_width=True, key="reset_sim"):
            st.session_state.current_index = 0
            st.session_state.transactions_processed = 0
            st.session_state.frauds_detected = 0
            st.session_state.latest_amount = 0.0
            st.session_state.simulator_running = False
    
    # calc metrics
    fraud_rate = (st.session_state.frauds_detected / max(st.session_state.transactions_processed, 1) * 100)
    
    with col_stat1:
        st.metric("Processed", st.session_state.transactions_processed)
    
    with col_stat2:
        st.metric("Frauds", st.session_state.frauds_detected, 
                 delta=f"{fraud_rate:.1f}%",
                 delta_color="inverse")
                 
    with col_stat3:
        st.metric("Last Amount", f"${st.session_state.latest_amount:.2f}")
    
    
    # Realtime Simulator Logic
    if st.session_state.simulator_running:
        
        # Check if we've reached the end of the dataset
        if st.session_state.current_index >= len(df):
            st.session_state.simulator_running = False
            st.success(f"Simulation complete! Processed all {len(df)} transactions.")
            st.rerun()
        
        row = df.iloc[st.session_state.current_index]
        
        # Run prediction
        try:
            # Prepare payload
            payload = {f"V{i}": row[f"V{i}"] for i in range(1, 29)}
            payload["Amount"] = row["Amount"]
            
            # Make prediction
            response = requests.post(f"{API_BASE_URL}/api/predict", json=payload, timeout=5)
            response.raise_for_status()
            result = response.json()
            
            prediction = result['Class']
            
            # Update statistics
            st.session_state.transactions_processed += 1
            st.session_state.latest_amount = row["Amount"]
            if prediction == "1":
                st.session_state.frauds_detected += 1
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
        
        # Move to next transaction
        st.session_state.current_index += 1
        
        # Wait 1 second between transactions
        time.sleep(1.0)
        
        # Force refresh to continue simulation
        st.rerun()
    
        st.rerun()
    
    st.divider()

    # Recent Transactions
    display_recent_transactions()

    st.divider()


    # ------------------------------------------------------------------------
    # ANOMALY EXPLANATION
    # ------------------------------------------------------------------------
    st.subheader("Explainable AI")
    
    col_exp1, col_exp2 = st.columns([1, 2])
    
    with col_exp1:
        # Filter mostly for fraud to be interesting
        recent_frauds = fetch_transactions(limit=50)
        
        if recent_frauds:
            selected_txn_id = st.selectbox(
                "Select Transaction", 
                options=[t['id'] for t in recent_frauds],
                format_func=lambda x: f"Transaction #{x}"
            )
            
            # Find the selected transaction data
            selected_txn = next((t for t in recent_frauds if t['id'] == selected_txn_id), None)
            
            if selected_txn and st.button("Explain Prediction", type="primary", use_container_width=True):
                # Prepare payload (remove ID/Class, keep features)
                payload = {k: v for k, v in selected_txn.items() if k in [f"V{i}" for i in range(1, 29)] + ["Amount"]}
                
                with st.spinner("Calculating SHAP values..."):
                    explanation = fetch_explanation(payload)
                    
                if explanation:
                    st.session_state.current_explanation = explanation
                    st.session_state.explained_txn_id = selected_txn_id
        else:
            st.warning("No recent fraud transactions found to explain.")

    with col_exp2:
        if 'current_explanation' in st.session_state:
            exp = st.session_state.current_explanation
            
            base_val = exp['base_value']
            contribs = exp['contributions']
            
            # Prepare data for Waterfall
            # Sort by absolute contribution, take top 10
            top_contribs = sorted(contribs, key=lambda x: abs(x['contribution']), reverse=True)[:10]
            
            feats = [c['feature'] for c in top_contribs]
            vals = [c['contribution'] for c in top_contribs]
            text_vals = [f"{c['value']:.2f}" for c in top_contribs]
            
            # Add "Other" if needed, but waterfall is tricky with "Other". 
            # Let's just show top 10 impactful features.
            
            fig_waterfall = go.Figure(go.Waterfall(
                orientation = "h",
                measure = ["relative"] * len(feats),
                x = vals,
                y = feats,
                text = text_vals,
                textposition = "outside",
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
            ))
            
            fig_waterfall.update_layout(
                title=f"Feature Contributions (Base Value: {base_val:.2f})",
                template='plotly_dark',
                height=500,
                xaxis_title="SHAP Value (Impact on Log-Odds)",
                yaxis_title="Feature"
            )
            
            st.plotly_chart(fig_waterfall, use_container_width=True)


def render_data_analytics(limit):
    """Page 2: Data Analytics"""
    
    # KPI Section
    st.title("Data Intelligence")
    st.subheader("Transaction Anomaly Monitoring & Analysis")
    
    stats = fetch_stats()
    display_metrics(stats)
    
    st.divider()

    # Distribution Analysis
    st.subheader("Transaction Pattern Analysis")
    
    amount_dist = fetch_amount_distribution(limit=limit)
    
    if amount_dist:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fraud = px.histogram(
                x=amount_dist['fraud_values'],
                nbins=50,
                title="Fraudulent Transaction Amounts",
                labels={'x': 'Amount ($)', 'y': 'Frequency'},
                color_discrete_sequence=['#ff4b4b']
            )
            fig_fraud.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_fraud, use_container_width=True)
        
        with col2:
            fig_normal = px.histogram(
                x=amount_dist['normal_values'],
                nbins=50,
                title="Legitimate Transaction Amounts",
                labels={'x': 'Amount ($)', 'y': 'Frequency'},
                color_discrete_sequence=['#00cc88']
            )
            fig_normal.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_normal, use_container_width=True)

    # Advanced Feature Analysis
    st.subheader("Deep Dive: Feature Correlations")
    
    tab1, tab2 = st.tabs(["Feature Histograms", "Scatter Relationships"])
    
    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_feature = st.selectbox(
                "Select PCA Feature", 
                ["Amount"] + [f"V{i}" for i in range(1, 29)],
                index=0,
                key="hist_feat_select"
            )
        
        dist_data = fetch_feature_distribution(selected_feature, limit)
        
        if dist_data:
            c1, c2 = st.columns(2)
            with c1:
                fig_f = px.histogram(
                    x=dist_data['fraud_values'], nbins=50, title=f"Fraud - {selected_feature}",
                    labels={'x': selected_feature, 'y': 'Frequency'}, color_discrete_sequence=['#ff4b4b']
                )
                fig_f.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig_f, use_container_width=True)
            with c2:
                fig_n = px.histogram(
                    x=dist_data['normal_values'], nbins=50, title=f"Normal - {selected_feature}",
                    labels={'x': selected_feature, 'y': 'Frequency'}, color_discrete_sequence=['#00cc88']
                )
                fig_n.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig_n, use_container_width=True)
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            x_feat = st.selectbox("X Axis", [f"V{i}" for i in range(1, 29)] + ["Amount"], index=0, key="scatter_x")
        with c2:
            y_feat = st.selectbox("Y Axis", [f"V{i}" for i in range(1, 29)] + ["Amount"], index=1, key="scatter_y")
            
        scatter_data = fetch_scatter_data(x_feat, y_feat, limit=limit)
        if scatter_data and scatter_data.get('points'):
            df_scatter = pd.DataFrame(scatter_data['points'])
            df_scatter['Type'] = df_scatter['is_fraud'].map({True: 'Fraud', False: 'Normal'})
            
            fig_scatter = px.scatter(
                df_scatter, x='x', y='y', color='Type',
                title=f"Correlation: {x_feat} vs {y_feat}",
                color_discrete_map={'Fraud': '#ff4b4b', 'Normal': '#00cc88'},
                opacity=0.7
            )
            fig_scatter.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # Recent Transactions
    display_recent_transactions()

def render_model_analytics(limit):
    """Page 3: Model Analytics"""
    st.title("Model Performance")
    
    # Autoencoder Training Progress
    st.subheader("🧠 Autoencoder Training Progress")
    st.markdown("**Training on 227,451 normal transactions**")
    
    # Create training progress data
    training_data = {
        'Epoch': [10, 20, 30, 40, 50],
        'Avg Loss': [0.6354, 0.5834, 0.5612, 0.5514, 0.5454],
        'Learning Rate': [0.000050, 0.000025, 0.000013, 0.000006, 0.000003]
    }
    df_training = pd.DataFrame(training_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Loss curve
        fig_loss = px.line(
            df_training, x='Epoch', y='Avg Loss',
            title="Training Loss Progression",
            markers=True,
            color_discrete_sequence=['#00cc88']
        )
        fig_loss.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig_loss, use_container_width=True)
    
    with col2:
        # Learning rate curve
        fig_lr = px.line(
            df_training, x='Epoch', y='Learning Rate',
            title="Learning Rate Schedule",
            markers=True,
            color_discrete_sequence=['#4c78a8']
        )
        fig_lr.update_layout(template='plotly_dark', height=350, yaxis_type="log")
        st.plotly_chart(fig_lr, use_container_width=True)
    
    st.divider()
    
    # XGBoost Training Info
    st.subheader("⚡ XGBoost Ensemble Training")
    col1, col2, col3 = st.columns(3)
    col1.metric("Normal Samples", "227,451")
    col2.metric("Fraud Samples", "394")
    col3.metric("Scale Weight", "577.29")
        
    st.divider()
    
    # Model Evaluation
    st.subheader("📈 Model Evaluation Results")
    
    # Metrics from classification report
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "99.96%")
    col2.metric("Precision (Fraud)", "92%")
    col3.metric("Recall (Fraud)", "85%")
    col4.metric("F1 Score (Fraud)", "88%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        # Actual confusion matrix from training output
        cm_data = [[56857, 7], [15, 83]]
        
        fig_cm = px.imshow(
            cm_data, 
            x=['Normal', 'Fraud'], 
            y=['Normal', 'Fraud'], 
            text_auto=True, 
            color_continuous_scale='Blues',
            labels=dict(x="Predicted Class", y="Actual Class", color="Count")
        )
        fig_cm.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Add interpretation
        st.markdown("""
        **Matrix Breakdown:**
        - True Negatives (TN): 56,857
        - False Positives (FP): 7
        - False Negatives (FN): 15
        - True Positives (TP): 83
        """)
    
    with col2:
        st.subheader("Classification Report")
        
        # Create classification report dataframe
        report_data = {
            'Class': ['Normal (0)', 'Fraud (1)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
            'Precision': [1.00, 0.92, None, 0.96, 1.00],
            'Recall': [1.00, 0.85, None, 0.92, 1.00],
            'F1-Score': [1.00, 0.88, 1.00, 0.94, 1.00],
            'Support': [56864, 98, 56962, 56962, 56962]
        }
        df_report = pd.DataFrame(report_data)
        
        st.dataframe(df_report, width="stretch", hide_index=True)
        
        st.markdown("""
        **Key Insights:**
        - Excellent overall accuracy (99.96%)
        - High precision for fraud detection (92%)
        - Good recall for fraud cases (85%)
        - Minimal false positives (only 7)
        """)


# ============================================================================
# Main Application Logic
# ============================================================================

def main():
    # Sidebar Navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Main Menu", ["Dashboard", "Data Intelligence", "Model Performance"], label_visibility="collapsed")
                            
        # Global Parameters
        limit = 5000 
        
        # Initialize session state for analysis button
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

    # Routing
    if page == "Dashboard":
        render_home()
    elif page == "Data Intelligence":
        render_data_analytics(limit)
    elif page == "Model Performance":
        render_model_analytics(limit)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>Fraud Detection Analytics v1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
