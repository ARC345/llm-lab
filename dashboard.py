
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

if glob.glob("runs/*"):
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
                                config.update(meta.get('args', {}))
                                
                # Add config params to df
                for k, v in config.items():
                    if isinstance(v, (list, dict)):
                        v = str(v)
                    df[f"config.{k}"] = v

                # Force numeric conversion for metrics
                numeric_cols = ['train_loss', 'val_loss', 'val_acc', 'ood_acc', 'dead_perc', 'grad_norm', 'act_mean', 'act_std', 'tokens_sec',
                               'intermediate_error', 'query_error', 'random_error', 'input_node_error']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                data.append(df)

                # Load Probe Data
                probe_results = []
                probing_dir = os.path.join(d, "probing")
                if os.path.isdir(probing_dir):
                    layer_dirs = glob.glob(os.path.join(probing_dir, "layer_*"))
                    for ld in layer_dirs:
                        try:
                            res_file = os.path.join(ld, "results.json")
                            if os.path.exists(res_file):
                                with open(res_file, 'r') as f:
                                    res = json.load(f)
                                    probe_results.append(res)
                        except:
                            pass
                
                if probe_results:
                    for pr in probe_results:
                        pr['run_name'] = run_name
                    
                    if 'probe_data_global' not in st.session_state:
                         st.session_state.probe_data_global = []
                    
                    # Simple check: remove existing entries for this run before adding
                    st.session_state.probe_data_global = [p for p in st.session_state.probe_data_global if p['run_name'] != run_name]
                    st.session_state.probe_data_global.extend(probe_results)

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

st.markdown("---")
st.header("Reasoning Metrics")

reasoning_cols = ['val_acc', 'ood_acc']
error_cols = ['intermediate_error', 'query_error', 'random_error', 'input_node_error']

if any(c in filtered_df.columns for c in reasoning_cols):
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        st.subheader("Reasoning Accuracy")
        if 'val_acc' in filtered_df.columns:
             fig_acc = px.line(
                filtered_df.dropna(subset=['val_acc']), 
                x='step', y='val_acc', color='run_name',
                title="2-Hop Test Accuracy"
            )
             st.plotly_chart(fig_acc)
             
    with col_r2:
        st.subheader("OOD Generalization")
        if 'ood_acc' in filtered_df.columns:
             fig_ood = px.line(
                filtered_df.dropna(subset=['ood_acc']), 
                x='step', y='ood_acc', color='run_name',
                title="3-Hop OOD Accuracy"
            )
             st.plotly_chart(fig_ood)

    # Error Analysis
    if any(c in filtered_df.columns for c in error_cols):
        st.subheader("Error Analysis (Recent)")
        latest_metrics = []
        for run in selected_experiments:
            run_df = filtered_df[filtered_df['run_name'] == run]
            # Find last row with valid error metrics
            valid_err_df = run_df.dropna(subset=error_cols, how='all')
            
            if not valid_err_df.empty:
                last_row = valid_err_df.iloc[-1]
                for err in error_cols:
                    if err in last_row:
                        latest_metrics.append({
                            'run': run,
                            'error_type': err,
                            'value': last_row[err]
                        })
        
        if latest_metrics:
            err_df = pd.DataFrame(latest_metrics)
            fig_err = px.bar(
                err_df, x='run', y='value', color='error_type',
                title="Error Distribution (Latest Step)",
                barmode='stack'
            )
            st.plotly_chart(fig_err)

st.markdown("---")
st.header("Internal Representation Analysis (Probing)")

if 'probe_data_global' not in st.session_state:
    st.session_state.probe_data_global = []

filtered_probe_data = [p for p in st.session_state.probe_data_global if p['run_name'] in selected_experiments]

if filtered_probe_data:
    probe_df = pd.DataFrame(filtered_probe_data)
    if 'layer' in probe_df.columns and 'accuracy' in probe_df.columns:
        probe_df['layer'] = pd.to_numeric(probe_df['layer'])
        fig_probe = px.line(
            probe_df.sort_values(by='layer'), 
            x='layer', y='accuracy', color='run_name',
            title="Probe Accuracy by Layer (Linear Classification of Intermediate Node)",
            markers=True,
            hover_data=['samples', 'checkpoint']
        )
        st.plotly_chart(fig_probe)
else:
    st.info("No probing results found for selected runs.")

# 4. Summary Table
st.subheader("Final Metrics Summary")
summary = []
for run in selected_experiments:
    run_df = all_metrics[all_metrics['run_name'] == run]
    if run_df.empty: continue
    
    last_row = run_df.iloc[-1]
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
