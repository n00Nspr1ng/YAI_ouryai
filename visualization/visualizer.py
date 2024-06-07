from typing import *
import os

import pickle
import blosc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tap


# for 3d_diffuser_actor data format
def visualize_dat_file(file_path: str) -> None: # .dat file
    def read_dat_file(file_path: str):
        with open(file_path, 'rb') as f:
            compressed_data = f.read()
            data = pickle.loads(blosc.decompress(compressed_data))
        return data

    data = read_dat_file(file_path)
    keyframe_num = len(data[0])

    # Split into RGB and pcd data
    rgb_data = data[1][:, :, 0]  # (keyframes, cameras, 3, H, W)
    pcd_data = data[1][:, :, 1]  # (keyframes, cameras, 3, H, W)
    cam_num = rgb_data.shape[1]

    # Trajectory of the gripper
    gripper_poses = data[5]

    # Initialize Dash app
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(id='2d-images', style={'display': 'inline-block', 'width': '48%'}),
        dcc.Graph(id='3d-scatter', style={'display': 'inline-block', 'width': '48%'}),
        dcc.Slider(
            id='keyframe-slider',
            min=0,
            max=keyframe_num - 1,
            value=0,
            marks={i: str(i + 1) for i in range(keyframe_num)}
        )
    ])

    @app.callback(
        [Output('2d-images', 'figure'),
         Output('3d-scatter', 'figure')],
        [Input('keyframe-slider', 'value')]
    )
    def update_figure(keyframe_idx):
        fig_2d = make_subplots(
            rows=2, cols=2, 
            subplot_titles=['Left Shoulder', 'Right Shoulder', 'Wrist', 'Front']
        )

        for cam_idx in range(cam_num):
            row = cam_idx // 2 + 1
            col = cam_idx % 2 + 1

            # RGB Image
            rgb_image = rgb_data[keyframe_idx, cam_idx].transpose(1, 2, 0)
            rgb_image = (rgb_image - (-1)) / (1 - (-1)) * 256  # Normalize to [0, 256]
            fig_2d.add_trace(go.Image(z=rgb_image), row=row, col=col)

        fig_2d.update_layout(height=600, width=600, title_text=f'Keyframe {keyframe_idx + 1} - RGB Images')
        
        # 3D Scene - only front view
        pcd_points = None
        colors = None
        for cam_idx in range(cam_num):
            pcd_point = pcd_data[keyframe_idx, cam_idx].reshape(3, -1).T
            color = rgb_data[keyframe_idx, cam_idx].reshape(3, -1).T
            color = (color - (-1)) / (1 - (-1))

            if pcd_points is None:
                pcd_points = pcd_point
                colors = color
            else:
                pcd_points = np.concatenate((pcd_points, pcd_point), axis=0)
                colors = np.concatenate((colors, color), axis=0)

        filtered_pcd = pcd_points[(pcd_points[:, 0] > -1.0) & (pcd_points[:, 2] > 0.5)]
        filtered_colors = colors[(pcd_points[:, 0] > -1.0) & (pcd_points[:, 2] > 0.5)]

        scatter3d = go.Scatter3d(
            x=filtered_pcd[:, 0],
            y=filtered_pcd[:, 1],
            z=filtered_pcd[:, 2],
            mode='markers',
            marker=dict(size=2, color=filtered_colors)
        )

        fig_3d = go.Figure(data=[scatter3d])

        # Gripper trajectory
        for poses in gripper_poses:
            for pose in poses:
                x, y, z, _, _, _, _, gripper_state = pose
                color = 'blue' if gripper_state == 1 else 'red' # Blue marker if the gripper is open, else red
                fig_3d.add_trace(
                    go.Scatter3d(
                        x=[x], y=[y], z=[z],
                        mode='markers',
                        marker=dict(size=3, color=color)
                    )
                )

        fig_3d.update_layout(
            title=f'Keyframe {keyframe_idx + 1} - 3D Scene',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
            )
        )

        return fig_2d, fig_3d

    app.run_server(debug=True)


def visualize_denoising_process(args) -> None:

    def read_file(file_path: str):
        # diff_traj
        with open(os.path.join(file_path, 'diff_traj.pkl'), 'rb') as f:
            diff_traj = pickle.load(f)
        # outputs
        with open(os.path.join(file_path, 'outputs.pkl'), 'rb') as f:
            outputs = pickle.load(f)
        
        return diff_traj, outputs
    
    diff_traj, outputs = read_file(file_path=args.file_path) 
    keypose_num = len(outputs['pointcloud'])

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    while True:
        print()
        print(f"Type which keypose you want to visualize (between 0 and {keypose_num - 1}). Type -1 to quit.")
        keypose_idx = int(input())

        if keypose_idx == -1:
            break

        # Fixed points
        fixed_pcd = outputs['pointcloud'][5]
        fixed_color = outputs['colors'][5] / 255
        ax.scatter(
            fixed_pcd[:, 0], 
            fixed_pcd[:, 1], 
            fixed_pcd[:, 2], 
            marker='.', 
            c=fixed_color,
            s=1
        )

        points = [ax.plot([], [], [], 'o', c='blue')[0] for _ in range(diff_traj[keypose_idx].shape[0])]

        def init():
            for point in points:
                point.set_data([], [])
                point.set_3d_properties([])
            return points

        def update(frame):
            for i, point in enumerate(points):
                x = diff_traj[1][i, frame, 0]
                y = diff_traj[1][i, frame, 1]
                z = diff_traj[1][i, frame, 2]
                point.set_data(x, y)
                point.set_3d_properties(z + 0.05)
            return points
        
        # Animation
        ani = animation.FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=50)
        
        plt.show()


class Arguments(tap.Tap):
    type: str = "dataset" # "dataset" or "denoising"
    file_path: str = ""


if __name__ == "__main__":

    # Arguments
    args = Arguments().parse_args()

    if args.type == "dataset":
        visualize_dat_file(args.file_path)
    elif args.type == "denoising":
        visualize_denoising_process(args)
