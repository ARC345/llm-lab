from clearml import Task
import os
from datetime import datetime

POSTS_DIR = "posts"

def generate_blog_post_from_task(task):
    if not task:
        print("No task provided.")
        return

    # Fetch data from task
    task_id = task.id
    task_name = task.name
    started_at = task.data.started
    
    if not started_at:
        # Fallback if task hasn't "started" or date is missing
        date_obj = datetime.now()
    else:
        date_obj = started_at

    title_date = date_obj.strftime('%B %d, %Y')
    filename_date = date_obj.strftime('%Y-%m-%d')
    
    # Get configuration (settings class)
    # ClearML stores class parameters in 'General' or other sections depending on how it was connected
    # We'll look for the user properties or hyperparams
    params = task.get_parameters_as_dict()
    
    # Organize params into a table
    config_table = "| Parameter | Value |\n| :--- | :--- |\n"
    # Flatten checks: usually comes as {'General': {'key': 'value'}}
    for section, values in params.items():
        if isinstance(values, dict):
            for k, v in values.items():
                config_table += f"| `{k}` | `{v}` |\n"
        else:
             config_table += f"| `{section}` | `{values}` |\n"

    # Get Metrics: Best Val Loss
    # We look for the scalar 'Loss' / 'val'
    scalars = task.get_reported_scalars()
    best_val_loss = "N/A"
    if scalars and 'Loss' in scalars and 'val' in scalars['Loss']:
        # scalars['Loss']['val']['y'] is a list of values
        val_losses = scalars['Loss']['val']['y']
        if val_losses:
            best_val_loss = f"{min(val_losses):.4f}"

    # Get Generated Text
    # We typically log this via report_text. 
    # Unfortunately fetching reported text via API can be tricky, it's stored in events.
    # For simplicity, we might skipping full text fetching or use a workaround if needed.
    # But let's try to get the last reported text if possible, or just link to the dashboard.
    dashboard_url = task.get_output_log_web_page()
    
    content = f"""---
title: "Experiment: {task_name}"
date: {filename_date}
categories: [experiment, machine-learning]
tags: [gpt, clearml, training]
message: "{task.comment or 'No comment provided'}"
---

# Experiment Summary
**Task ID**: `{task_id}`
**Date**: {title_date}
**Best Val Loss**: `{best_val_loss}`
[View on ClearML Dashboard]({dashboard_url})

## Hyperparameters
{config_table}

## Loss Curve
*(Check the dashboard for interactive plots)*
"""

    post_filename = f"{filename_date}-clearml-{task_id}.md"
    post_path = os.path.join(POSTS_DIR, post_filename)
    os.makedirs(POSTS_DIR, exist_ok=True)

    with open(post_path, "w") as f:
        f.write(content)
    
    print(f"Created blog post: {post_path}")

if __name__ == "__main__":
    # Get the most recent completed task from this project
    tasks = Task.get_tasks(
        project_name="gpt-from-scratch", 
        task_name="train_run",
        task_filter={'status': ['completed', 'published']} 
    )
    
    if not tasks:
        print("No completed ClearML tasks found for project 'gpt-from-scratch'.")
        # Fallback: try to get current running task? No, only completed.
    else:
        # Sort by creation date descending
        tasks.sort(key=lambda t: t.data.created, reverse=True)
        latest_task = tasks[0]
        print(f"Generating blog post for latest task: {latest_task.id}")
        generate_blog_post_from_task(latest_task)

