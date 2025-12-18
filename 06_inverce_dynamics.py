import numpy as np
from simulator import Simulator
from pathlib import Path
from typing import Dict
import os 
import pinocchio as pin

print(pin.__version__)

def task_space_controller(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict) -> np.ndarray:
    """Task space inverse dynamics controller."""

    # Update forward kinematics
    pin.framesForwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)

    # ID фрейма end-effector
    ee_frame_id = model.getFrameId("end_effector")

    # Current pose end-effector
    current_se3 = data.oMf[ee_frame_id]

    # Desired pose
    desired_position = desired['pos']
    desired_quaternion = desired['quat']  # [w, x, y, z] MuJoCo
    desired_quaternion_pin = np.array([*desired_quaternion[1:], desired_quaternion[0]])  # [x,y,z,w]
    desired_pose = np.concatenate([desired_position, desired_quaternion_pin])
    desired_se3 = pin.XYZQUATToSE3(desired_pose)

    # Error task space
    pose_error = desired_se3.actInv(current_se3)  # M_err = M_des^{-1} * M_current
    spatial_velocity_error = pin.log6(pose_error).vector

    # Real spatial velocity end-effector (V = J * dq)
    J = pin.getFrameJacobian(model, data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)  # 6xnv, в world
    ee_velocity = J @ dq  # 6D

    # Coefficients for PD in task space
    kp_task = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    kd_task = np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0])

    # Goal speed in task space
    desired_acc_task = kp_task * (-spatial_velocity_error) - kd_task * ee_velocity

    # Inv Jacobian
    lambda_damping = 1e-4
    J_pinv = np.linalg.pinv(J, rcond=lambda_damping)

    # Goal acceleration in joint space
    ddq_des = J_pinv @ (desired_acc_task - pin.getFrameJacobianTimeVariation(model, data, ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED) @ dq)

    # τ = M * ddq_des + coriolis + gravity
    nonlinear_effects = pin.rnea(model, data, q, dq, np.zeros_like(q))
    
    log_metrics(q, dq, t, desired, ddq_des, desired_acc_task)

    # M * ddq_des
    tau = nonlinear_effects + pin.rnea(model, data, q, dq, ddq_des) - nonlinear_effects
    
    return tau
    
import csv
from datetime import datetime

log_time = []
log_pos = []      
log_euler = []     
log_task_vel = []   
log_task_acc = []    
log_joint_pos = []    
log_joint_vel = []  
log_joint_acc = []
log_desired_pos = []
log_desired_euler = []

def log_metrics(q: np.ndarray, dq: np.ndarray, t: float, desired: Dict, ddq_des: np.ndarray, desired_acc_task: np.ndarray):

    pin.framesForwardKinematics(model, data, q)
    ee_frame_id = model.getFrameId("end_effector")
    current_se3 = data.oMf[ee_frame_id]
    current_position = current_se3.translation.copy()

    current_rotation = current_se3.rotation.copy()
    euler_angles = pin.rpy.matrixToRpy(current_rotation).tolist()

    J = pin.getFrameJacobian(model, data, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    task_vel = (J @ dq).tolist()

    desired_pos = desired['pos'].tolist()

    desired_quat_mujoco = np.array(desired['quat'])
    desired_quat_pin = np.roll(desired_quat_mujoco, -1)
    desired_rotation = pin.Quaternion(desired_quat_pin).matrix()
    desired_euler = pin.rpy.matrixToRpy(desired_rotation).tolist()

    log_time.append(t)
    log_pos.append(current_position.tolist())
    log_euler.append(euler_angles)
    log_task_vel.append(task_vel)
    log_task_acc.append(desired_acc_task.tolist())
    log_joint_pos.append(q.tolist())
    log_joint_vel.append(dq.tolist())
    log_joint_acc.append(ddq_des.tolist())
    log_desired_pos.append(desired_pos)
    log_desired_euler.append(desired_euler)

def save_logs_to_csv(filename: str = None):
    if filename is None:
        filename = f"robot_metrics_06_task_space.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['time']
        headers += [f'pos_{coord}' for coord in ['x', 'y', 'z']]
        headers += [f'euler_{angle}' for angle in ['roll', 'pitch', 'yaw']]
        headers += [f'task_vel_{i}' for i in range(1, 7)]  
        headers += [f'task_acc_{i}' for i in range(1, 7)] 
        headers += [f'joint_pos_{j}' for j in range(1, 7)]
        headers += [f'joint_vel_{j}' for j in range(1, 7)] 
        headers += [f'joint_acc_{j}' for j in range(1, 7)] 
        headers += [f'desired_pos_{coord}' for coord in ['x', 'y', 'z']]
        headers += [f'desired_euler_{angle}' for angle in ['roll', 'pitch', 'yaw']]
        writer.writerow(headers)
        
        for i in range(len(log_time)):
            row = [log_time[i]]
            row.extend(log_pos[i])     
            row.extend(log_euler[i])     
            row.extend(log_task_vel[i])  
            row.extend(log_task_acc[i])  
            row.extend(log_joint_pos[i])
            row.extend(log_joint_vel[i]) 
            row.extend(log_joint_acc[i])
            row.extend(log_desired_pos[i]) 
            row.extend(log_desired_euler[i])
            writer.writerow(row)
    
    print(f"Logs saved in {filename}")

def main():
    # Create logging directories
    Path("logs/videos").mkdir(parents=True, exist_ok=True)
    
    print("\nRunning task space controller...")
    sim = Simulator(
        xml_path="robots/universal_robots_ur5e/scene.xml",
        enable_task_space=True,
        show_viewer=True,
        record_video=True,
        video_path="logs/videos/06_task_space.mp4",
        fps=30,
        width=1920,
        height=1080
    )
    sim.set_controller(task_space_controller)
    sim.run(time_limit=60.0)
    save_logs_to_csv()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "robots/universal_robots_ur5e/ur5e.xml")
    model = pin.buildModelFromMJCF(xml_path)
    data = model.createData()
    main() 
