import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from datetime import datetime, timedelta
import io
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import threading
import queue

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Cell Management System",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2563eb, #1e40af);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563eb;
        margin: 0.5rem 0;
    }
    
    .cell-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .cell-card.active {
        border-left: 4px solid #059669;
        background: #f0fdf4;
    }
    
    .task-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 0.5rem 0;
    }
    
    .task-card.active {
        border-left: 4px solid #059669;
        background: #f0fdf4;
    }
    
    .task-card.completed {
        border-left: 4px solid #d97706;
        background: #fffbeb;
    }
    
    .status-running {
        color: #059669;
        font-weight: bold;
    }
    
    .status-stopped {
        color: #dc2626;
        font-weight: bold;
    }
    
    .stSelectbox > div > div {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class Cell:
    id: str
    type: str
    voltage: float
    min_voltage: float
    max_voltage: float
    current: float
    temperature: float
    capacity: float
    is_active: bool = False
    health: float = 100.0
    cycle_count: int = 0

@dataclass
class Task:
    id: str
    type: str
    duration: int
    current: float
    voltage: float = 0.0
    status: str = "pending"  # pending, active, completed
    progress: float = 0.0
    start_time: datetime = None
    end_time: datetime = None

class CellManagementSystem:
    def __init__(self):
        self.cells: List[Cell] = []
        self.tasks: List[Task] = []
        self.data_log: List[Dict] = []
        self.is_running: bool = False
        self.current_task_index: int = 0
        self.start_time: datetime = None
        self.simulation_thread = None
        self.data_queue = queue.Queue()
        
    def create_cell(self, cell_id: int, cell_type: str) -> Cell:
        """Create a cell with specifications based on type"""
        cell_specs = {
            'lfp': {'voltage': 3.2, 'min_voltage': 2.8, 'max_voltage': 3.6},
            'li-ion': {'voltage': 3.7, 'min_voltage': 3.2, 'max_voltage': 4.2},
            'nimh': {'voltage': 1.2, 'min_voltage': 1.0, 'max_voltage': 1.4},
            'lead-acid': {'voltage': 2.1, 'min_voltage': 1.8, 'max_voltage': 2.4}
        }
        
        specs = cell_specs.get(cell_type, cell_specs['li-ion'])
        
        return Cell(
            id=f"cell_{cell_id}_{cell_type}",
            type=cell_type,
            voltage=specs['voltage'],
            min_voltage=specs['min_voltage'],
            max_voltage=specs['max_voltage'],
            current=0.0,
            temperature=round(random.uniform(25, 40), 1),
            capacity=0.0
        )
    
    def add_task(self, task_type: str, duration: int, current: float, voltage: float = 0.0):
        """Add a new task to the system"""
        task = Task(
            id=f"task_{len(self.tasks) + 1}",
            type=task_type,
            duration=duration,
            current=current,
            voltage=voltage
        )
        self.tasks.append(task)
        return task
    
    def update_cell_parameters(self, task: Task, elapsed_time: float):
        """Update cell parameters based on current task"""
        for cell in self.cells:
            if not cell.is_active:
                continue
                
            # Base voltage for the cell type
            base_voltage = cell.voltage
            variation = np.sin(elapsed_time / 10) * 0.1 + (random.random() - 0.5) * 0.05
            
            if task.type == "CC_CV":
                # Charging: voltage increases
                progress_ratio = elapsed_time / task.duration
                cell.voltage = min(cell.max_voltage, 
                                 base_voltage + progress_ratio * 0.5 + variation)
            elif task.type == "CC_CD":
                # Discharging: voltage decreases
                progress_ratio = elapsed_time / task.duration
                cell.voltage = max(cell.min_voltage, 
                                 base_voltage - progress_ratio * 0.3 + variation)
            elif task.type == "IDLE":
                # Idle: voltage stable with small variations
                cell.voltage = base_voltage + variation * 0.1
            
            # Update current based on task
            cell.current = task.current * (0.8 + random.random() * 0.4)
            
            # Update temperature (increases during active periods)
            base_temp = 25 + random.random() * 15
            cell.temperature = base_temp + (5 if cell.is_active else 0)
            
            # Update capacity
            cell.capacity = cell.voltage * cell.current
            
            # Simulate battery health degradation (very slow)
            if cell.is_active:
                cell.health = max(0, cell.health - 0.001)
    
    def log_data_point(self, task: Task, elapsed_time: float):
        """Log current system state"""
        timestamp = datetime.now()
        
        for cell in self.cells:
            data_point = {
                'timestamp': timestamp,
                'task_id': task.id,
                'task_type': task.type,
                'task_elapsed': round(elapsed_time, 1),
                'cell_id': cell.id,
                'cell_type': cell.type,
                'voltage': round(cell.voltage, 3),
                'current': round(cell.current, 3),
                'temperature': round(cell.temperature, 1),
                'capacity': round(cell.capacity, 3),
                'is_active': cell.is_active,
                'health': round(cell.health, 1)
            }
            self.data_log.append(data_point)
    
    def simulation_worker(self):
        """Background simulation worker"""
        while self.is_running and self.current_task_index < len(self.tasks):
            current_task = self.tasks[self.current_task_index]
            
            # Initialize task if not started
            if current_task.status == "pending":
                current_task.status = "active"
                current_task.start_time = datetime.now()
                
                # Activate cells
                for cell in self.cells:
                    cell.is_active = True
            
            # Calculate elapsed time
            elapsed = (datetime.now() - current_task.start_time).total_seconds()
            current_task.progress = min(100, (elapsed / current_task.duration) * 100)
            
            # Update cell parameters
            self.update_cell_parameters(current_task, elapsed)
            
            # Log data
            self.log_data_point(current_task, elapsed)
            
            # Put update signal in queue
            self.data_queue.put("update")
            
            # Check if task is completed
            if elapsed >= current_task.duration:
                current_task.status = "completed"
                current_task.end_time = datetime.now()
                current_task.progress = 100
                
                # Deactivate cells
                for cell in self.cells:
                    cell.is_active = False
                    cell.current = 0
                
                self.current_task_index += 1
            
            time.sleep(1)  # Update every second
        
        # Simulation completed
        self.is_running = False
        for cell in self.cells:
            cell.is_active = False
            cell.current = 0
    
    def start_simulation(self):
        """Start the simulation"""
        if len(self.tasks) == 0:
            return False, "No tasks configured"
        
        self.is_running = True
        self.start_time = datetime.now()
        self.current_task_index = 0
        
        # Reset task statuses
        for task in self.tasks:
            task.status = "pending"
            task.progress = 0
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self.simulation_worker)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        return True, "Simulation started successfully"
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        
        # Reset states
        for task in self.tasks:
            if task.status == "active":
                task.status = "pending"
                task.progress = 0
        
        for cell in self.cells:
            cell.is_active = False
            cell.current = 0
        
        self.current_task_index = 0
    
    def export_data_csv(self) -> str:
        """Export logged data to CSV format"""
        if not self.data_log:
            return ""
        
        df = pd.DataFrame(self.data_log)
        
        # Convert timestamp to string for CSV
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

# Initialize session state
if 'cms' not in st.session_state:
    st.session_state.cms = CellManagementSystem()

# Get the system instance
cms = st.session_state.cms

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔋 Advanced Cell Management System</h1>
    <p>Real-time Battery Cell Monitoring & Task Management</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # Cell Configuration
    st.subheader("🔋 Cell Setup")
    num_cells = st.number_input("Number of Cells", min_value=1, max_value=20, value=4, key="num_cells")
    
    # Cell type configuration
    if st.button("Configure Cells") or len(cms.cells) != num_cells:
        cms.cells = []
        for i in range(num_cells):
            cell_type = st.selectbox(
                f"Cell {i+1} Type", 
                ["lfp", "li-ion", "nimh", "lead-acid"],
                key=f"cell_type_{i}",
                index=i % 2  # Alternate between types
            )
            cms.cells.append(cms.create_cell(i+1, cell_type))
    
    st.divider()
    
    # Task Management
    st.subheader("📋 Task Management")
    
    task_type = st.selectbox(
        "Task Type",
        ["CC_CV", "IDLE", "CC_CD"],
        help="CC_CV: Constant Current/Voltage, IDLE: Rest, CC_CD: Constant Current Discharge"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Duration (s)", min_value=1, value=60)
    with col2:
        current = st.number_input("Current (A)", min_value=0.1, value=1.5, step=0.1)
    
    if task_type in ["CC_CV", "CC_CD"]:
        voltage = st.number_input("Voltage (V)", min_value=0.1, value=3.7, step=0.1)
    else:
        voltage = 0.0
    
    if st.button("➕ Add Task"):
        cms.add_task(task_type, duration, current, voltage)
        st.success(f"Task {task_type} added successfully!")
    
    st.divider()
    
    # Simulation Controls
    st.subheader("🎮 Simulation Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", disabled=cms.is_running):
            success, message = cms.start_simulation()
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with col2:
        if st.button("⏹️ Stop", disabled=not cms.is_running):
            cms.stop_simulation()
            st.info("Simulation stopped")
    
    # System Status
    status = "🟢 RUNNING" if cms.is_running else "🔴 STOPPED"
    st.markdown(f"**Status:** {status}")
    
    if cms.is_running and cms.current_task_index < len(cms.tasks):
        current_task = cms.tasks[cms.current_task_index]
        st.progress(current_task.progress / 100)
        st.caption(f"Task: {current_task.type} ({current_task.progress:.1f}%)")
    
    st.divider()
    
    # Data Export
    st.subheader("📊 Data Export")
    st.info(f"📈 {len(cms.data_log)} data points logged")
    
    if st.button("💾 Export CSV") and cms.data_log:
        csv_data = cms.export_data_csv()
        st.download_button(
            label="Download CSV Report",
            data=csv_data,
            file_name=f"cell_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Main dashboard
col1, col2, col3, col4 = st.columns(4)

# Dashboard metrics
active_cells = sum(1 for cell in cms.cells if cell.is_active)
avg_voltage = np.mean([cell.voltage for cell in cms.cells]) if cms.cells else 0
total_current = sum(cell.current for cell in cms.cells)
avg_temp = np.mean([cell.temperature for cell in cms.cells]) if cms.cells else 0

with col1:
    st.metric("🔋 Active Cells", active_cells, delta=None)

with col2:
    st.metric("⚡ Avg Voltage", f"{avg_voltage:.2f}V", delta=None)

with col3:
    st.metric("🔌 Total Current", f"{total_current:.2f}A", delta=None)

with col4:
    st.metric("🌡️ Avg Temperature", f"{avg_temp:.1f}°C", delta=None)

# Cell Status Grid
st.subheader("🔋 Cell Status Overview")

if cms.cells:
    # Create columns for cell display
    cols = st.columns(min(4, len(cms.cells)))
    
    for i, cell in enumerate(cms.cells):
        with cols[i % 4]:
            status_class = "active" if cell.is_active else ""
            cell_info = f"""
            <div class="cell-card {status_class}">
                <h4>Cell {cell.id.split('_')[1]}</h4>
                <p><strong>Type:</strong> {cell.type.upper()}</p>
                <p><strong>Voltage:</strong> {cell.voltage:.2f}V</p>
                <p><strong>Current:</strong> {cell.current:.2f}A</p>
                <p><strong>Temp:</strong> {cell.temperature:.1f}°C</p>
                <p><strong>Health:</strong> {cell.health:.1f}%</p>
            </div>
            """
            st.markdown(cell_info, unsafe_allow_html=True)

# Task List
st.subheader("📋 Task Queue")

if cms.tasks:
    for i, task in enumerate(cms.tasks):
        status_class = ""
        if task.status == "active":
            status_class = "active"
        elif task.status == "completed":
            status_class = "completed"
        
        task_info = f"""
        <div class="task-card {status_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4>Task {i+1}: {task.type}</h4>
                    <p>Duration: {task.duration}s | Current: {task.current}A</p>
                </div>
                <div>
                    <p><strong>{task.status.upper()}</strong></p>
                    <p>{task.progress:.1f}%</p>
                </div>
            </div>
        </div>
        """
        st.markdown(task_info, unsafe_allow_html=True)
else:
    st.info("No tasks configured. Add tasks using the sidebar.")

# Real-time Charts
if cms.data_log:
    st.subheader("📈 Real-time Monitoring")
    
    # Convert data to DataFrame
    df = pd.DataFrame(cms.data_log)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get recent data (last 100 points per cell)
    recent_df = df.groupby('cell_id').tail(100).reset_index(drop=True)
    
    tab1, tab2, tab3 = st.tabs(["Voltage Trends", "Current Analysis", "Temperature Monitor"])
    
    with tab1:
        # Voltage chart
        fig_voltage = px.line(
            recent_df, 
            x='timestamp', 
            y='voltage', 
            color='cell_id',
            title='Real-time Voltage Monitoring',
            labels={'voltage': 'Voltage (V)', 'timestamp': 'Time'}
        )
        fig_voltage.update_layout(height=400)
        st.plotly_chart(fig_voltage, use_container_width=True)
    
    with tab2:
        # Current chart
        fig_current = px.line(
            recent_df, 
            x='timestamp', 
            y='current', 
            color='cell_id',
            title='Real-time Current Monitoring',
            labels={'current': 'Current (A)', 'timestamp': 'Time'}
        )
        fig_current.update_layout(height=400)
        st.plotly_chart(fig_current, use_container_width=True)
    
    with tab3:
        # Temperature chart
        fig_temp = px.line(
            recent_df, 
            x='timestamp', 
            y='temperature', 
            color='cell_id',
            title='Real-time Temperature Monitoring',
            labels={'temperature': 'Temperature (°C)', 'timestamp': 'Time'}
        )
        fig_temp.update_layout(height=400)
        st.plotly_chart(fig_temp, use_container_width=True)
    
    # System overview charts
    st.subheader("📊 System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cell type distribution
        cell_types = [cell.type for cell in cms.cells]
        type_counts = pd.Series(cell_types).value_counts()
        
        fig_pie = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Cell Type Distribution"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Task status distribution
        if cms.tasks:
            task_statuses = [task.status for task in cms.tasks]
            status_counts = pd.Series(task_statuses).value_counts()
            
            fig_bar = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                title="Task Status Distribution",
                labels={'x': 'Status', 'y': 'Count'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

# Auto-refresh functionality
if cms.is_running:
    # Check for updates from simulation thread
    try:
        cms.data_queue.get_nowait()
        st.rerun()
    except queue.Empty:
        pass
    
    # Auto-refresh every 2 seconds when running
    time.sleep(2)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7;">
    <p>🔋 Advanced Cell Management System | Built with Streamlit</p>
    <p>Real-time monitoring • Interactive visualizations • CSV export</p>
</div>
""", unsafe_allow_html=True)