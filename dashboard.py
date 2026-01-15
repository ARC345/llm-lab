
import streamlit as st
import pandas as pd
import json
import os
import glob
import plotly.express as px

st.set_page_config(layout="wide", page_title="Experiment Dashboard")

st.title("GPT Experiment Dashboard")

# 1. Load Data
run_dirs = glob.glob("runs/*")
data = []

for d in run_dirs:
    run_name = os.path.basename(d)
    metrics_path = os.path.join(d, "metrics.csv")
    meta_path = os.path.join(d, "meta.jsonl")
    
    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            df['run_name'] = run_name
            
            # Load config from meta
            config = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    for line in f:
                        meta = json.loads(line)
                        if meta.get('type') == 'config':
                            # Update config, allowing later overrides (like resumes) to take precedence
                            # or we can keep initial. Let's just grab the last one.
                            config.update(meta.get('args', {}))
                            
            # Add config params to df (for filtering if we wanted, but mainly for display)
            for k, v in config.items():
                df[f"config.{k}"] = v
                
            data.append(df)
        except Exception as e:
            st.warning(f"Failed to load {run_name}: {e}")

if not data:
    st.warning("No experiment data found in 'runs/'.")
    st.stop()

all_metrics = pd.concat(data, ignore_index=True)

# 2. Sidebar Controls
st.sidebar.header("Filter Experiments")
selected_experiments = st.sidebar.multiselect(
    "Select Runs", 
    options=all_metrics['run_name'].unique(),
    default=all_metrics['run_name'].unique()
)

if not selected_experiments:
    st.stop()

filtered_df = all_metrics[all_metrics['run_name'].isin(selected_experiments)]

# 3. Visualizations

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Loss")
    fig_loss = px.line(
        filtered_df.dropna(subset=['train_loss']), 
        x='step', y='train_loss', color='run_name',
        title="Training Loss", hover_data=['lr', 'tokens_sec']
    )
    st.plotly_chart(fig_loss)

with col2:
    st.subheader("Validation Loss")
    fig_val_loss = px.line(
        filtered_df.dropna(subset=['val_loss']), 
        x='step', y='val_loss', color='run_name',
        title="Validation Loss"
    )
    st.plotly_chart(fig_val_loss)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Tokens / Second")
    fig_tps = px.line(
        filtered_df, 
        x='step', y='tokens_sec', color='run_name',
        title="Throughput (tok/sec)"
    )
    st.plotly_chart(fig_tps)

with col4:
    st.subheader("Dead Neurons %")
    # Some experiments might not have this metric
    if 'dead_perc' in filtered_df.columns:
        fig_dead = px.line(
            filtered_df, 
            x='step', y='dead_perc', color='run_name',
            title="Dead Neurons Percentage"
        )
        st.plotly_chart(fig_dead)
    else:
        st.info("No dead neuron tracking data found.")

st.markdown("---")
st.header("Advanced Metrics")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Gradient Norm")
    if 'grad_norm' in filtered_df.columns:
        fig_grad = px.line(
            filtered_df, 
            x='step', y='grad_norm', color='run_name',
            title="Global Gradient Norm"
        )
        st.plotly_chart(fig_grad)
    else:
        st.info("No gradient norm data found.")

with col6:
    st.subheader("Activation Mean")
    if 'act_mean' in filtered_df.columns:
        fig_act_mean = px.line(
            filtered_df, 
            x='step', y='act_mean', color='run_name',
            title="Activation Mean"
        )
        st.plotly_chart(fig_act_mean)
    else:
        st.info("No activation mean data found.")

col7, col8 = st.columns(2)

with col7:
    st.subheader("Activation Std Dev")
    if 'act_std' in filtered_df.columns:
        fig_act_std = px.line(
            filtered_df, 
            x='step', y='act_std', color='run_name',
            title="Activation Std Dev"
        )
        st.plotly_chart(fig_act_std)
    else:
        st.info("No activation std data found.")

# 4. Summary Table
st.subheader("Final Metrics Summary")
summary = []
for run in selected_experiments:
    run_df = all_metrics[all_metrics['run_name'] == run]
    if run_df.empty: continue
    
    last_row = run_df.iloc[-1]
    
    # Get best val loss
    best_val_loss = run_df['val_loss'].min()
    
    summary.append({
        "Run": run,
        "Steps": last_row['step'],
        "Best Val Loss": best_val_loss,
        "Final Train Loss": last_row['train_loss'] if pd.notna(last_row['train_loss']) else "N/A",
        "Learning Rate": last_row['config.learning_rate'],
        "Model": last_row.get('config.experiment', 'Unknown')
    })

st.dataframe(pd.DataFrame(summary))
