import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse

def plot_metrics(files_list):

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan']

    dfs = []
    labels = []
    
    for i, file in enumerate(files_list):
        df = pd.read_csv(file)
        dfs.append(df)
        labels.append(file.split('/')[-1])
    
    # Graf 1: pose (x, y, z)
    fig_pos = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=('Position X', 'Position Y', 'Position Z'))
    
    for row, coord in enumerate(['x', 'y', 'z'], start=1):
        for i, df in enumerate(dfs):
            fig_pos.add_trace(go.Scatter(x=df['time'], y=df[f'pos_{coord}'],
                                         mode='lines', name=f"{labels[i]}",
                                         line=dict(color=colors[i])), row=row, col=1)

            fig_pos.add_trace(go.Scatter(x=df['time'], y=df[f'desired_pos_{coord}'],
                                         mode='lines', name=f"{labels[i]} Desired",
                                         line=dict(color='black', dash='dash')), row=row, col=1)
    
    fig_pos.update_layout(title='Position in Time', xaxis3_title='Time (s)', height=900)
    fig_pos.show()
    
    # Graf 2: Euler (roll, pitch, yaw) in time
    fig_euler = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              subplot_titles=('Euler Roll', 'Euler Pitch', 'Euler Yaw'))
    
    for row, angle in enumerate(['roll', 'pitch', 'yaw'], start=1):
        for i, df in enumerate(dfs):
            fig_euler.add_trace(go.Scatter(x=df['time'], y=df[f'euler_{angle}'],
                                           mode='lines', name=f"{labels[i]}",
                                           line=dict(color=colors[i])), row=row, col=1)
                                           
            fig_euler.add_trace(go.Scatter(x=df['time'], y=df[f'desired_euler_{angle}'],
                                           mode='lines', name=f"{labels[i]} Desired",
                                           line=dict(color='black', dash='dash')), row=row, col=1)
    
    fig_euler.update_layout(title='Euler Angles in Time', xaxis3_title='Time (s)', height=900)
    fig_euler.show()
    
    # Graf 3: Task Velocity (J1-J6) in time
    fig_task_vel = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                 subplot_titles=[f'Task Velocity {j}' for j in range(1,7)])
    
    for row in range(1, 7):
        for i, df in enumerate(dfs):
            fig_task_vel.add_trace(go.Scatter(x=df['time'], y=df[f'task_vel_{row}'],
                                              mode='lines', name=f"{labels[i]}",
                                              line=dict(color=colors[i])), row=row, col=1)
    
    fig_task_vel.update_layout(title='Task Velocity in Time', xaxis6_title='Time (s)', height=1200)
    fig_task_vel.show()
    
    # Graf 4: Task Acceleration (J1-J6) in time
    fig_task_acc = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                 subplot_titles=[f'Task Acceleration {j}' for j in range(1,7)])
    
    for row in range(1, 7):
        for i, df in enumerate(dfs):
            fig_task_acc.add_trace(go.Scatter(x=df['time'], y=df[f'task_acc_{row}'],
                                              mode='lines', name=f"{labels[i]}",
                                              line=dict(color=colors[i])), row=row, col=1)
    
    fig_task_acc.update_layout(title='Task Acceleration in Time', xaxis6_title='Time (s)', height=1200)
    fig_task_acc.show()
    
    # Graf 5: Joint Positions (J1-J6) in time
    fig_joint_pos = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                  subplot_titles=[f'Joint Position {j}' for j in range(1,7)])
    
    for row in range(1, 7):
        for i, df in enumerate(dfs):
            fig_joint_pos.add_trace(go.Scatter(x=df['time'], y=df[f'joint_pos_{row}'],
                                               mode='lines', name=f"{labels[i]}",
                                               line=dict(color=colors[i])), row=row, col=1)
    
    fig_joint_pos.update_layout(title='Joint Positions in Time', xaxis6_title='Time (s)', height=1200)
    fig_joint_pos.show()
    
    # Graf 6: Joint Velocities (J1-J6) in time
    fig_joint_vel = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                  subplot_titles=[f'Joint Velocity {j}' for j in range(1,7)])
    
    for row in range(1, 7):
        for i, df in enumerate(dfs):
            fig_joint_vel.add_trace(go.Scatter(x=df['time'], y=df[f'joint_vel_{row}'],
                                               mode='lines', name=f"{labels[i]}",
                                               line=dict(color=colors[i])), row=row, col=1)
    
    fig_joint_vel.update_layout(title='Joint Velocities in Time', xaxis6_title='Time (s)', height=1200)
    fig_joint_vel.show()
    
    # Graf 7: Joint Accelerations (J1-J6) in time
    fig_joint_acc = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                  subplot_titles=[f'Joint Acceleration {j}' for j in range(1,7)])
    
    for row in range(1, 7):
        for i, df in enumerate(dfs):
            fig_joint_acc.add_trace(go.Scatter(x=df['time'], y=df[f'joint_acc_{row}'],
                                               mode='lines', name=f"{labels[i]}",
                                               line=dict(color=colors[i])), row=row, col=1)
    
    fig_joint_acc.update_layout(title='Joint Accelerations in Time', xaxis6_title='Time (s)', height=1200)
    fig_joint_acc.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot robot metrics from CSV files")
    parser.add_argument('files', nargs='+', help="List of CSV files to compare")
    args = parser.parse_args()
    
    plot_metrics(args.files)
