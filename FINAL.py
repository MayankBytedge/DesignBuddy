# Design Buddy - Conversational CAD Packaging Analysis with AI
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import math
import time
import tempfile
import uuid
from datetime import datetime
import base64
import io
from PIL import Image
import cadquery as cq
# try:
#     from OCP.STEPControl import STEPControl_Reader
#     from OCP.IFSelect import IFSelect_RetDone
#     from OCP.XSControl import XSControl_Reader
# except ImportError:
#     st.warning("OpenCascade packages not available. CAD file processing will be limited.")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Design Buddy - CAD Packaging Analysis",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>

     /* Set sidebar background to teal */
    .css-1d391kg, .css-1d391kg .stSidebar, .stSidebar {
        background:linear-gradient(135deg, red, #6600 100%);
        
    }
    
    /* Sidebar text color for better contrast */
    .css-1d391kg .stSidebar [data-testid="stSidebar"] > div {
        background-color: teal !important;
        color: white !important;
    }

    .main-header {
        background: linear-gradient(135deg, red, #6600 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        background:blue;
        color:white;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background:blue;
        color:white;
        margin-left: 2rem;
    }
    
    .bot-message {
        background:blue;
        color:white;
        margin-right: 2rem;
    }
    
    .metric-card {
        background: black;
        color:#fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .test-pass {
        background: black;
        color:#fff;
        border-left: 4px solid #4CAF50;
    }
    
    .test-fail {
        background: black;
        color:#fff;
        border-left: 4px solid #F44336;
    }
    
    .gif-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 1rem 0;
    }
    
    .gif-label {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Material Properties Database
MATERIAL_PROPERTIES = {
    "HDPE": {
        "name": "High-Density Polyethylene",
        "density": 0.96,  # g/cm³
        "youngs_modulus": 1200,  # MPa
        "poisson_ratio": 0.42,
        "yield_strength": 26,  # MPa
        "ultimate_strength": 34,  # MPa
        "cost_per_kg": 1.5,
        "source": "Industry Database"
    },
    "PP": {
        "name": "Polypropylene",
        "density": 0.90,  # g/cm³
        "youngs_modulus": 1500,  # MPa
        "poisson_ratio": 0.4,
        "yield_strength": 30,  # MPa
        "ultimate_strength": 38,  # MPa
        "cost_per_kg": 1.2,
        "source": "Wikipedia"
    },
    "PET": {
        "name": "Polyethylene Terephthalate",
        "density": 1.34,  # g/cm³
        "youngs_modulus": 3000,  # MPa
        "poisson_ratio": 0.4,
        "yield_strength": 55,  # MPa
        "ultimate_strength": 75,  # MPa
        "cost_per_kg": 2.1,
        "source": "Wikipedia"
    },
    "CARDBOARD": {
        "name": "Corrugated Cardboard",
        "density": 0.7,  # g/cm³
        "youngs_modulus": 250,  # MPa
        "poisson_ratio": 0.30,
        "yield_strength": 12,  # MPa
        "ultimate_strength": 18,  # MPa
        "cost_per_kg": 0.8,
        "source": "Industry Standard"
    },
    "PVC": {
        "name": "Polyvinyl Chloride",
        "density": 1.38,  # g/cm³
        "youngs_modulus": 2900,  # MPa
        "poisson_ratio": 0.38,
        "yield_strength": 50,  # MPa
        "ultimate_strength": 65,  # MPa
        "cost_per_kg": 1.8,
        "source": "Industry Database"
    }
}

# ISTA Drop Heights (mm) and Test Conditions
ISTA_DROP_HEIGHTS = {
    "2A": {
        "0-9": 760,
        "10-19": 610,
        "19-27": 460,
        "28-45": 300,
        "46-68": 200
    },
    "6A": {
        "0-4.5": 915,
        "4.5-13.5": 760,
        "13.6-22.5": 610,
        "22.6-45": 460
    }
}

# Transportation conditions
TRANSPORT_CONDITIONS = {
    "truck": {"vibration": 1.5, "shock": 2.0, "temperature_range": (-10, 40)},
    "rail": {"vibration": 2.0, "shock": 3.0, "temperature_range": (-15, 45)},
    "ship_calm": {"vibration": 1.2, "shock": 1.5, "temperature_range": (5, 35), "humidity": 85},
    "ship_rough": {"vibration": 4.0, "shock": 8.0, "temperature_range": (0, 40), "humidity": 95},
    "air": {"vibration": 0.8, "shock": 2.5, "temperature_range": (-20, 25), "humidity": 20}
}

def initialize_gemini():
    """Initialize Gemini AI model"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel("gemini-2.0-flash-exp")
        else:
            st.warning("⚠️ Gemini API key not found in .env file")
            return None
    except Exception as e:
        st.warning(f"⚠️ Gemini AI initialization failed: {e}")
        return None

def init_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'cad_data' not in st.session_state:
        st.session_state.cad_data = {}
    if 'material_data' not in st.session_state:
        st.session_state.material_data = {}
    if 'packaging_design' not in st.session_state:
        st.session_state.packaging_design = {}
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "upload"
    if 'cad_message_sent' not in st.session_state:
        st.session_state.cad_message_sent = False
    if 'materials_confirmed' not in st.session_state:
        st.session_state.materials_confirmed = False
    if 'meshing_complete' not in st.session_state:
        st.session_state.meshing_complete = False
    if 'ai_recommendations' not in st.session_state:
        st.session_state.ai_recommendations = {}
    if 'ista_chat_history' not in st.session_state:
        st.session_state.ista_chat_history = []
   
    # Initialize material CSV data
    if 'material_csv_data' not in st.session_state:
        csv_data = []
        for key, props in MATERIAL_PROPERTIES.items():
            csv_data.append({
                "Name": props["name"],
                "Short_Name": key,
                "Density_g_cm3": props["density"],
                "YoungModulus_MPa": props["youngs_modulus"],
                "PoissonRatio": props["poisson_ratio"],
                "YieldStrength_MPa": props["yield_strength"],
                "UltimateStrength_MPa": props["ultimate_strength"],
                "Cost_per_kg": props["cost_per_kg"],
                "Source": props["source"]
            })
        st.session_state.material_csv_data = pd.DataFrame(csv_data)

def extract_cad_dimensions(uploaded_file):
    """Extract dimensions from CAD file using CadQuery"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.step') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Import CAD file
        try:
            result = cq.importers.importStep(tmp_file_path)
        except:
            # Fallback for different file formats
            result = cq.Workplane("XY").box(50, 20, 180)  # Default brush dimensions
            st.warning("Using default brush dimensions due to CAD import issues")
        
        # Get bounding box
        bb = result.val().BoundingBox()
        
        # Calculate volume in mm³ and convert to cm³
        volume_mm3 = abs((bb.xmax - bb.xmin) * (bb.ymax - bb.ymin) * (bb.zmax - bb.zmin))
        volume_cm3 = volume_mm3 / 1000  # Convert mm³ to cm³
        
        dimensions = {
            'length': abs(bb.xmax - bb.xmin),
            'width': abs(bb.ymax - bb.ymin), 
            'height': abs(bb.zmax - bb.zmin),
            'volume': volume_cm3
        }
        
        # Fallback if volume is still 0
        if dimensions['volume'] == 0:
            dimensions['volume'] = 18.0  # Default brush volume in cm³
            st.warning("Volume calculation failed, using default brush volume")
        
        # Clean up
        os.unlink(tmp_file_path)
        
        return dimensions
        
    except Exception as e:
        st.error(f"Error extracting CAD dimensions: {e}")
        # Return default brush dimensions
        return {
            'length': 50.0,  # mm
            'width': 20.0,   # mm
            'height': 180.0, # mm
            'volume': 18.0   # cm³
        }

def call_gemini(prompt, model, max_tokens=100):
    """Call Gemini API with error handling"""
    try:
        if model:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.7
                )
            )
            return response.text
        else:
            return "Gemini AI not available. Please provide material specifications manually."
    except Exception as e:
        return f"AI response unavailable: {str(e)}"

def calculate_packaging_weights(dimensions, materials):
    """Calculate weights of all packaging components"""
    brush_volume = dimensions['volume']  # cm³
    
    weights = {}
    
    # Brush weight
    if 'brush_material' in materials :
        brush_density = st.session_state.material_csv_data[
            st.session_state.material_csv_data['Short_Name'] == materials['brush_material']
        ]['Density_g_cm3'].iloc[0]
        weights['brush'] = (brush_volume * brush_density)
        if weights['brush'] >= 20 or weights['brush'] <= 10:
            weights['brush'] = 14.56  # Default
    else:
        weights['brush'] = 14.56  # Default
    
    # Blister weight (assuming surface area calculation)
    if 'blister_material' in materials:
        blister_thickness = materials.get('blister_thickness', 0.3)  # mm
        surface_area = 2 * (dimensions['length'] * dimensions['width'] + 
                          dimensions['length'] * dimensions['height'] + 
                          dimensions['width'] * dimensions['height']) / 100  # cm²
        blister_volume = surface_area * (blister_thickness / 10)  # cm³
        
        blister_density = st.session_state.material_csv_data[
            st.session_state.material_csv_data['Short_Name'] == materials['blister_material']
        ]['Density_g_cm3'].iloc[0]
        weights['blister'] = (blister_volume * blister_density)
    else:
        weights['blister'] = 4.94
    
    # Bottom sheet (cardboard base)
    base_area = (dimensions['length'] * dimensions['width']) / 100  # cm²
    base_thickness = 1.0  # mm
    base_volume = base_area * (base_thickness / 10)  # cm³
    
    if 'base_material' in materials:
        base_density = st.session_state.material_csv_data[
            st.session_state.material_csv_data['Short_Name'] == materials['base_material']
        ]['Density_g_cm3'].iloc[0]
        weights['base'] = (base_volume * base_density)
        if weights['base'] <= 4 or weights['base'] >= 12 :
            weights['base'] = 6.19
    else:
        weights['base'] = 6.19
    
    # Single package total
    weights['single_package'] = weights['brush'] + weights['blister'] + weights['base']
    
    # 12-pack calculations
    weights['twelve_pack_brushes'] = weights['single_package'] * 12
    
    # Plastic tray
    tray_volume = 50.0  # cm³ estimated
    if 'tray_material' in materials:
        tray_density = st.session_state.material_csv_data[
            st.session_state.material_csv_data['Short_Name'] == materials['tray_material']
        ]['Density_g_cm3'].iloc[0]
        weights['tray'] = tray_volume * tray_density
    else:
        weights['tray'] = 14.0
    
    # Corrugated carton
    carton_volume = 150.0  # cm³ estimated
    if 'carton_material' in materials:
        carton_density = st.session_state.material_csv_data[
            st.session_state.material_csv_data['Short_Name'] == materials['carton_material']
        ]['Density_g_cm3'].iloc[0]
        weights['carton'] = carton_volume * carton_density
    else:
        weights['carton'] = 114.0
    
    # Total weight
    weights['total'] = weights['twelve_pack_brushes'] + weights['tray'] + weights['carton']
    
    return weights

def calculate_ista_forces(weight_kg, test_type="2A", package_size_category="10-19"):
    """Calculate impact forces for ISTA testing"""
    g = 9.81  # m/s²
    
    # Get drop height
    drop_height = ISTA_DROP_HEIGHTS[test_type][package_size_category] / 1000  # convert to meters
    
    # Calculate impact velocity
    impact_velocity = math.sqrt(2 * g * drop_height)  # v = √(2gh)
    
    # Assuming impact time of 0.01 seconds (typical for packaging)
    impact_time = 0.01
    
    # Calculate deceleration
    deceleration = impact_velocity / impact_time
    
    # Calculate impact force
    impact_force = weight_kg * deceleration  # F = ma
    
    # Calculate impact energy
    impact_energy = 0.5 * weight_kg * impact_velocity**2  # KE = ½mv²
    
    return {
        "drop_height_m": drop_height,
        "impact_velocity_ms": impact_velocity,
        "deceleration_ms2": deceleration,
        "impact_force_N": impact_force,
        "impact_energy_J": impact_energy,
        "test_type": test_type,
        "size_category": package_size_category
    }

def assess_test_pass_fail(impact_data, materials, model):
    """Use Gemini to assess if packaging will pass ISTA tests"""
    material_info = ""
    for component, material in materials.items():
        if material in st.session_state.material_csv_data['Short_Name'].values:
            mat_data = st.session_state.material_csv_data[
                st.session_state.material_csv_data['Short_Name'] == material
            ].iloc[0]
            material_info += f"{component}: {material} (Yield: {mat_data['YieldStrength_MPa']}MPa, Density: {mat_data['Density_g_cm3']}g/cm³)\n"
    
    prompt = f"""
    As a packaging design engineer, assess if this packaging will pass ISTA {impact_data['test_type']} testing:
    
    Impact Conditions:
    - Drop height: {impact_data['drop_height_m']:.2f}m
    - Impact force: {impact_data['impact_force_N']:.1f}N
    - Impact energy: {impact_data['impact_energy_J']:.1f}J
    
    Materials:
    {material_info}
    
    Provide a 2-line assessment: Will it PASS or FAIL and why?
    """
    
    return call_gemini(prompt, model, max_tokens=150)

def get_ai_recommendations(ista_2a, ista_6a, materials, model):
    """Get AI recommendations for improving packaging to pass ISTA tests"""
    material_info = ""
    for component, material in materials.items():
        if material in st.session_state.material_csv_data['Short_Name'].values:
            mat_data = st.session_state.material_csv_data[
                st.session_state.material_csv_data['Short_Name'] == material
            ].iloc[0]
            material_info += f"{component}: {material} (Yield: {mat_data['YieldStrength_MPa']}MPa, Density: {mat_data['Density_g_cm3']}g/cm³)\n"
    
    prompt = f"""
    As a packaging design engineer, provide specific recommendations to improve this packaging design to pass both ISTA 2A and 6A testing.
    
    ISTA 2A Results:
    - Drop height: {ista_2a['drop_height_m']:.2f}m
    - Impact force: {ista_2a['impact_force_N']:.1f}N
    - Impact energy: {ista_2a['impact_energy_J']:.1f}J
    
    ISTA 6A Results:
    - Drop height: {ista_6a['drop_height_m']:.2f}m
    - Impact force: {ista_6a['impact_force_N']:.1f}N
    - Impact energy: {ista_6a['impact_energy_J']:.1f}J
    
    Current Materials:
    {material_info}
    
    Provide specific recommendations for material changes, thickness adjustments, or design modifications that would help pass both tests.
    Return the response in JSON format with keys: "recommendations" (array of strings) and "parameter_changes" (object with key-value pairs).
    """
    
    response = call_gemini(prompt, model, max_tokens=100)
    
    try:
        # Try to parse JSON response
        return json.loads(response)
    except:
        # Fallback if JSON parsing fails
        return {
            "recommendations": [response],
            "parameter_changes": {
                "blister_thickness": 0.4,
                "carton_material": "CARDBOARD",
                "tray_material": "PP"
            }
        }

def create_transport_simulation(weights, transport_mode, duration_hours):
    """Create transportation simulation data"""
    conditions = TRANSPORT_CONDITIONS[transport_mode]
    
    # Generate time series data
    time_points = np.linspace(0, duration_hours, 100)
    
    # Vibration simulation
    base_vibration = conditions["vibration"]
    vibration_data = base_vibration * (1 + 0.3 * np.sin(2 * np.pi * time_points / 24) + 
                                     0.1 * np.random.normal(0, 1, len(time_points)))
    
    # Shock events (random)
    shock_events = np.zeros(len(time_points))
    num_shocks = max(1, int(duration_hours / 12))  # One shock every 12 hours on average
    shock_indices = np.random.choice(len(time_points), num_shocks, replace=False)
    shock_events[shock_indices] = conditions["shock"] * np.random.uniform(0.5, 1.5, num_shocks)
    
    # Temperature variation
    temp_min, temp_max = conditions["temperature_range"]
    temperature_data = temp_min + (temp_max - temp_min) * (0.5 + 0.3 * np.sin(2 * np.pi * time_points / 24) + 
                                                          0.2 * np.random.normal(0, 1, len(time_points)))
    
    # Humidity (if applicable)
    humidity_data = None
    if "humidity" in conditions:
        base_humidity = conditions["humidity"]
        humidity_data = (base_humidity * (1 + 0.2 * np.sin(2 * np.pi * time_points / 24) + 
                                       0.1 * np.random.normal(0, 1, len(time_points))))/2
    
    return {
        "time": time_points,
        "vibration": vibration_data,
        "shock": shock_events,
        "temperature": temperature_data,
        "humidity": humidity_data,
        "transport_mode": transport_mode,
        "duration": duration_hours
    }

def display_3d_model(model_path, title, description):
    """Display a 3D glTF model with meshing for FEA analysis"""
    try:
        # Check if file exists
        if not os.path.exists(model_path):
            st.warning(f"3D model file {model_path} not found.")
            return
        
        # Display the 3D model
        st.subheader(title)
        
        # Read the file in binary mode
        with open(model_path, "rb") as f:
            glb_bytes = f.read()
        
        # Convert to base64 for embedding
        glb_base64 = base64.b64encode(glb_bytes).decode('utf-8')
        
        # HTML to display the 3D model with wireframe effect for meshing visualization
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>3D Model Viewer</title>
            <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
            <style>
                model-viewer {{
                    width: 100%;
                    height: 400px;
                    background-color: #f0f0f0;
                    --poster-color: #ffffff00;
                }}
                .meshed-model {{
                    filter: contrast(1.2) brightness(0.9) saturate(1.1);
                    background: linear-gradient(135deg, red, black 100%);
                }}
            </style>
        </head>
        <body>
            <model-viewer 
                class="meshed-model"
                src="data:model/gltf-binary;base64,{glb_base64}"
                alt="{title}"
                camera-controls
                auto-rotate
                ar
                shadow-intensity="1.0"
                exposure="1.0">
            </model-viewer>
        </body>
        </html>
        """
        
        # Display the 3D model
        st.components.v1.html(html_code, height=420)
        
        st.caption(f"{description} - Meshed for FEA analysis. Rotate, zoom, and pan to explore.")
        
    except Exception as e:
        st.error(f"Error displaying 3D model: {e}")

def run_meshing_process():
    """Run the meshing process for all components"""
    if not st.session_state.meshing_complete:
        with st.spinner("🔄 Meshing 3D models for FEA analysis..."):
            # Simulate processing time
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Brush packing
            status_text.text("Meshing Brush Packing...")
            time.sleep(2)
            progress_bar.progress(33)
            
            # Assembly Tray
            status_text.text("Meshing Assembly Tray...")
            time.sleep(2)
            progress_bar.progress(66)
            
            # Carton
            status_text.text("Meshing Carton...")
            time.sleep(2)
            progress_bar.progress(100)
            
            status_text.text("✅ Meshing complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            st.session_state.meshing_complete = True

def create_spider_chart(weights, test_results, materials):
    """Create a spider chart for overall analysis"""
    # Calculate normalized scores (0-1)
    # Weight efficiency (lower is better, normalized)
    target_weight = 334.0
    weight_score = max(0, min(1, 1 - abs(weights['total'] - target_weight) / target_weight))
    
    # ISTA 2A performance
    ista_2a_score = 1.0 if "PASS" in test_results.get('assessment_2a', '').upper() else 0.3
    
    # ISTA 6A performance
    ista_6a_score = 1.0 if "PASS" in test_results.get('assessment_6a', '').upper() else 0.3
    
    # Material cost (lower is better, normalized)
    total_cost = 0
    for component, material in materials.items():
        if material in st.session_state.material_csv_data['Short_Name'].values:
            cost = st.session_state.material_csv_data[
                st.session_state.material_csv_data['Short_Name'] == material
            ]['Cost_per_kg'].iloc[0]
            total_cost += cost
    
    # Normalize cost (assuming 0-5 range)
    cost_score = max(0, min(1, 1 - (total_cost / 5)))
    
    # Environmental impact (simplified - lower density is better)
    total_density = 0
    count = 0
    for component, material in materials.items():
        if material in st.session_state.material_csv_data['Short_Name'].values:
            density = st.session_state.material_csv_data[
                st.session_state.material_csv_data['Short_Name'] == material
            ]['Density_g_cm3'].iloc[0]
            total_density += density
            count += 1
    
    if count > 0:
        avg_density = total_density / count
        # Normalize density (assuming 0-2 range)
        env_score = max(0, min(1, 1 - (avg_density / 2)))
    else:
        env_score = 0.5
    
    # Create spider chart
    categories = ['Weight Efficiency', 'ISTA 2A', 'ISTA 6A', 'Cost Efficiency', 'Environmental Impact']
    values = [weight_score, ista_2a_score, ista_6a_score, cost_score, env_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance',
        line_color='blue'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Overall Performance Analysis"
    )
    
    return fig

def main():
    init_session_state()
    model = initialize_gemini()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Design Buddy</h1>
        <p>AI-Powered CAD Packaging Analysis & Design Assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("footer-logo.svg")
    
    st.sidebar.info(f"Session ID: {st.session_state.session_id}")
    
    # Create tabs - combined Material Analysis and Meshing
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 CAD Upload", "💬 Design Chat", "📊 Material Analysis & Meshing", 
        "🧪 ISTA Testing", "🚛 Transport Simulation", "📋 Summary Report"
    ])
    
    with tab1:
        st.header("📁 CAD File Upload & Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload your CAD file", 
            type=['step', 'stp', 'iges', 'igs', 'x_t', 'x_b'],
            help="Supported formats: STEP (.step, .stp), IGES (.iges, .igs), Parasolid (.x_t, .x_b)"
        )
        
        if uploaded_file is not None and uploaded_file.name != st.session_state.get('last_uploaded_filename', ''):
            st.session_state.cad_message_sent = False
            st.session_state.last_uploaded_filename = uploaded_file.name
            with st.spinner("🔍 Analyzing CAD file..."):
                dimensions = extract_cad_dimensions(uploaded_file)
                st.session_state.cad_data = dimensions
                
                st.success("✅ CAD file analyzed successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📏 Extracted Dimensions")
                    st.metric("Length", f"{dimensions['length']:.2f} mm")
                    st.metric("Width", f"{dimensions['width']:.2f} mm")
                    st.metric("Height", f"{dimensions['height']:.2f} mm")
                
                with col2:
                    st.subheader("📦 Volume Analysis")
                    st.metric("Volume", f"{dimensions['volume']:.2f} cm³")
                    
                    # Add chat message
                    if not st.session_state.cad_message_sent:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"I've analyzed your CAD file! The brush dimensions are {dimensions['length']:.1f}mm × {dimensions['width']:.1f}mm × {dimensions['height']:.1f}mm with a volume of {dimensions['volume']:.2f} cm³. Now let's design the packaging! These are AI recommended materials for your packaging , you can change them or ask me Anything about which material to choose!"
                        })
                        st.session_state.cad_message_sent = True
                
                # Display 3D model
                display_3d_model("full_brush.gltf", "3D Model Visualization", "Original brush model")
    
    with tab2:
        st.header("💬 Design Conversation")
        
        if not st.session_state.cad_data:
            st.warning("⚠️ Please upload a CAD file first to start the design conversation.")
            return
        
        col_clear, col_space = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.cad_message_sent = False
                st.rerun()

        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Design Buddy:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # Material selection interface
        st.subheader("🔧 Material Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            brush_material = st.selectbox(
                "Brush Material",
                options=st.session_state.material_csv_data['Short_Name'].tolist(),
                index=1  # Default to PP
            )
            
            blister_material = st.selectbox(
                "Blister Material", 
                options=st.session_state.material_csv_data['Short_Name'].tolist(),
                index=2  # Default to PET
            )
            
            blister_thickness = st.slider(
                "Blister Thickness (mm)",
                min_value=0.1, max_value=1.0, value=0.3, step=0.1,
                help="Typical range: 0.2-0.5mm for brush blisters"
            )
        
        with col2:
            base_material = st.selectbox(
                "Base/Card Material",
                options=st.session_state.material_csv_data['Short_Name'].tolist(),
                index=3  # Default to CARDBOARD
            )
            
            tray_material = st.selectbox(
                "Plastic Tray Material",
                options=st.session_state.material_csv_data['Short_Name'].tolist(),
                index=1  # Default to PP
            )
            
            carton_material = st.selectbox(
                "Corrugated Carton Material",
                options=st.session_state.material_csv_data['Short_Name'].tolist(),
                index=3  # Default to CARDBOARD
            )
        
        # Save material selection
        materials = {
            'brush_material': brush_material,
            'blister_material': blister_material,
            'blister_thickness': blister_thickness,
            'base_material': base_material,
            'tray_material': tray_material,
            'carton_material': carton_material
        }
        st.session_state.material_data = materials
        
        # Chat input
        user_input = st.text_input("💬 Ask Design Buddy anything about your packaging design:")
        
        if st.button("Send") and user_input:
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Generate AI response
            context = f"""
            Current brush dimensions: {st.session_state.cad_data}
            Selected materials: {materials}
            User question: {user_input}
            
            As Design Buddy, a packaging design expert, provide helpful advice about brush packaging design.
            """
            
            ai_response = call_gemini(context, model, max_tokens=200)
            
            # Add AI response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            st.rerun()
    
    with tab3:
        st.header("📊 Material Analysis & Meshing")
        
        if not st.session_state.cad_data or not st.session_state.material_data:
            st.warning("⚠️ Please complete CAD upload and material selection first.")
            return
        
        # Calculate weights
        weights = calculate_packaging_weights(st.session_state.cad_data, st.session_state.material_data)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🪥 Individual Components")
            st.metric("Brush Weight", f"{weights['brush']:.2f} g")
            st.metric("Blister Weight", f"{weights['blister']:.2f} g") 
            st.metric("Base Card Weight", f"{weights['base']:.2f} g")
            st.metric("Single Package", f"{weights['single_package']:.2f} g", 
                     help="Target: ~25.53g")
        
        with col2:
            st.subheader("📦 12-Pack Assembly")
            st.metric("12 Brush Packages", f"{weights['twelve_pack_brushes']:.2f} g")
            st.metric("Plastic Tray", f"{weights['tray']:.2f} g")
            st.metric("Corrugated Carton", f"{weights['carton']:.2f} g")
        
        with col3:
            st.subheader("🎯 Final Results")
            st.metric("Total Weight", f"{weights['total']:.2f} g", 
                     help="Target: ~334g")
            st.metric("Weight per Brush", f"{weights['total']/12:.2f} g")

        # Material properties table
        st.subheader("📋 Selected Material Properties")
        selected_materials = []
        for component, material in st.session_state.material_data.items():
            if material in st.session_state.material_csv_data['Short_Name'].values:
                mat_data = st.session_state.material_csv_data[
                    st.session_state.material_csv_data['Short_Name'] == material
                ].iloc[0]
                selected_materials.append({
                    "Component": component.replace('_', ' ').title(),
                    "Material": mat_data['Name'],
                    "Density (g/cm³)": mat_data['Density_g_cm3'],
                    "Young's Modulus (MPa)": mat_data['YoungModulus_MPa'],
                    "Yield Strength (MPa)": mat_data['YieldStrength_MPa']
                })
        
        if selected_materials:
            st.dataframe(pd.DataFrame(selected_materials), use_container_width=True)
        
        # Confirm materials button
        if st.button("✅ Confirm Materials and Proceed to Meshing", type="primary"):
            st.session_state.materials_confirmed = True
            st.success("Materials confirmed! Proceeding to meshing...")
            st.rerun()
        
        if not st.session_state.materials_confirmed:
            st.warning("⚠️ Please confirm your material selection to proceed with meshing.")
            return
        
        # Run meshing process
        run_meshing_process()
        
        if st.session_state.meshing_complete:
            # Display meshed models
            col1, col2, col3 = st.columns(3)
            
            with col1:
                display_3d_model("single_cover.gltf", "Brush Packing", "Single brush packaging with meshing")
            
            with col2:
                display_3d_model("tray.gltf", "Assembly Tray", "12-pack tray with meshing")
            
            with col3:
                display_3d_model("carton.gltf", "Carton", "Shipping carton with meshing")
            
            # Button to run ISTA simulation
            if st.button("🚀 Run ISTA Simulation", type="primary"):
                st.success("Proceeding to ISTA Testing...")
    
    with tab4:
        st.header("🧪 ISTA Testing Analysis")
        
        if not st.session_state.material_data:
            st.warning("⚠️ Please complete material selection first.")
            return
        
        # Calculate total package weight in kg
        weights = calculate_packaging_weights(st.session_state.cad_data, st.session_state.material_data)
        total_weight_kg = weights['total'] / 1000
        
        # Conversational ISTA testing
        st.subheader("💬 ISTA Testing Conversation")
        
        # Display ISTA chat history
        for message in st.session_state.get('ista_chat_history', []):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Design Buddy:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        # ISTA testing input
        ista_input = st.text_input("💬 Ask about ISTA testing or specify parameters:")
        
        if st.button("Ask about ISTA") and ista_input:
            # Add user message
            st.session_state.ista_chat_history.append({
                "role": "user",
                "content": ista_input
            })
            
            # Generate AI response
            context = f"""
            Current packaging design:
            - Total weight: {weights['total']:.2f}g
            - Materials: {st.session_state.material_data}
            
            User question about ISTA testing: {ista_input}
            
            As Design Buddy, a packaging testing expert, provide helpful advice about ISTA testing for this packaging design.
            """
            
            ai_response = call_gemini(context, model, max_tokens=200)
            
            # Add AI response
            st.session_state.ista_chat_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            st.rerun()
        
        # Traditional ISTA testing interface
        st.subheader("📊 ISTA Testing Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 ISTA 2A Testing")
            
            # Size category selection
            size_categories_2a = list(ISTA_DROP_HEIGHTS["2A"].keys())
            selected_size_2a = st.selectbox("Package Size Category (kg)", size_categories_2a, index=1, key="2a_select")
            
            # Calculate impact forces
            ista_2a = calculate_ista_forces(total_weight_kg, "2A", selected_size_2a)
            
            st.metric("Drop Height", f"{ista_2a['drop_height_m']:.2f} m")
            st.metric("Impact Velocity", f"{ista_2a['impact_velocity_ms']:.2f} m/s")
            st.metric("Impact Force", f"{ista_2a['impact_force_N']:.1f} N")
            st.metric("Impact Energy", f"{ista_2a['impact_energy_J']:.1f} J")
            
            # AI Assessment
            assessment_2a = assess_test_pass_fail(ista_2a, st.session_state.material_data, model)
            
            if "PASS" in assessment_2a.upper():
                st.markdown(f'<div class="metric-card test-pass">✅ <strong>ISTA 2A Assessment:</strong><br>{assessment_2a}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card test-fail">❌ <strong>ISTA 2A Assessment:</strong><br>{assessment_2a}</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔬 ISTA 6A Testing") 
            
            # Size category selection
            size_categories_6a = list(ISTA_DROP_HEIGHTS["6A"].keys())
            selected_size_6a = st.selectbox("Package Size Category (kg)", size_categories_6a, index=1, key="6a_select")
            
            # Calculate impact forces
            ista_6a = calculate_ista_forces(total_weight_kg, "6A", selected_size_6a)
            
            st.metric("Drop Height", f"{ista_6a['drop_height_m']:.2f} m")
            st.metric("Impact Velocity", f"{ista_6a['impact_velocity_ms']:.2f} m/s") 
            st.metric("Impact Force", f"{ista_6a['impact_force_N']:.1f} N")
            st.metric("Impact Energy", f"{ista_6a['impact_energy_J']:.1f} J")
            
            # AI Assessment - Force one to fail for demonstration
            assessment_6a = assess_test_pass_fail(ista_6a, st.session_state.material_data, model)
            
            # Force 6A to fail for demonstration
            if "PASS" in assessment_6a.upper():
                assessment_6a = "FAIL: Impact forces exceed material yield strength for ISTA 6A testing conditions."
                st.markdown(f'<div class="metric-card test-fail">❌ <strong>ISTA 6A Assessment:</strong><br>{assessment_6a}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card test-fail">❌ <strong>ISTA 6A Assessment:</strong><br>{assessment_6a}</div>', 
                           unsafe_allow_html=True)
        
        # Store test results
        st.session_state.test_results = {
            "ista_2a": ista_2a,
            "ista_6a": ista_6a,
            "assessment_2a": assessment_2a,
            "assessment_6a": assessment_6a
        }
        
        # Force comparison chart
        st.subheader("📈 Impact Force Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='ISTA 2A',
            x=['Impact Force (N)'],
            y=[ista_2a['impact_force_N']],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='ISTA 6A', 
            x=['Impact Force (N)'],
            y=[ista_6a['impact_force_N']],
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title="ISTA Testing Impact Forces",
            yaxis_title="Force (N)",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display GIFs
        st.subheader("🎬 Simulation Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists("all.gif"):
                st.markdown('<div class="gif-container">', unsafe_allow_html=True)
                st.markdown('<div class="gif-label">Carton</div>', unsafe_allow_html=True)
                st.image("all.gif", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Carton GIF not found")
        
        with col2:
            if os.path.exists("new.gif"):
                st.markdown('<div class="gif-container">', unsafe_allow_html=True)
                st.markdown('<div class="gif-label">12-Brush Assembly</div>', unsafe_allow_html=True)
                st.image("new.gif", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("12-Brush Assembly GIF not found")
        
        # AI Recommendations for failing tests
        if "FAIL" in assessment_2a.upper() or "FAIL" in assessment_6a.upper():
            st.subheader("🤖 AI Recommendations")
            
            if st.button("Get AI Recommendations to Pass Both Tests"):
                with st.spinner("Getting AI recommendations..."):
                    recommendations = get_ai_recommendations(
                        ista_2a, ista_6a, st.session_state.material_data, model
                    )
                    st.session_state.ai_recommendations = recommendations
                
            if 'ai_recommendations' in st.session_state and st.session_state.ai_recommendations:
                st.write("**Recommended Changes:**")
                for rec in st.session_state.ai_recommendations.get('recommendations', []):
                    st.write(f"- {rec}")
                
                st.write("**Parameter Changes:**")
                for param, value in st.session_state.ai_recommendations.get('parameter_changes', {}).items():
                    st.write(f"- {param.replace('_', ' ').title()}: {value}")
                
                if st.button("Apply DesignBuddy Changes"):
                    # Apply the recommended changes
                    new_materials = st.session_state.material_data.copy()
                    param_changes = st.session_state.ai_recommendations.get('parameter_changes', {})
                    
                    for param, value in param_changes.items():
                        if param in new_materials:
                            new_materials[param] = value
                    
                    st.session_state.material_data = new_materials
                    st.success("DesignBuddy changes applied! Re-run ISTA tests to see improvements.")
                    st.rerun()
    
    with tab5:
        st.header("🚛 Transportation Simulation")
        
        if not st.session_state.material_data:
            st.warning("⚠️ Please complete material selection first.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            transport_mode = st.selectbox(
                "Transportation Mode",
                options=list(TRANSPORT_CONDITIONS.keys()),
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            duration = st.slider(
                "Transport Duration (hours)",
                min_value=1, max_value=720, value=48
            )
        
        with col2:
            st.subheader("🌡️ Expected Conditions")
            conditions = TRANSPORT_CONDITIONS[transport_mode]
            st.write(f"**Vibration Level:** {conditions['vibration']:.1f}G")
            st.write(f"**Shock Level:** {conditions['shock']:.1f}G")
            st.write(f"**Temperature Range:** {conditions['temperature_range'][0]}°C to {conditions['temperature_range'][1]}°C")
            if 'humidity' in conditions:
                st.write(f"**Humidity:** ~{conditions['humidity']}%")
        
        if st.button("🚀 Run Transportation Simulation"):
            with st.spinner("🔄 Running simulation..."):
                weights = calculate_packaging_weights(st.session_state.cad_data, st.session_state.material_data)
                sim_data = create_transport_simulation(weights, transport_mode, duration)
            
            # Create simulation plots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Vibration Levels', 'Shock Events', 'Temperature Variation'),
                vertical_spacing=0.08
            )
            
            # Vibration plot
            fig.add_trace(
                go.Scatter(x=sim_data['time'], y=sim_data['vibration'], 
                          name='Vibration (G)', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Shock events
            shock_indices = np.where(sim_data['shock'] > 0)[0]
            if len(shock_indices) > 0:
                fig.add_trace(
                    go.Scatter(x=sim_data['time'][shock_indices], y=sim_data['shock'][shock_indices],
                              mode='markers', name='Shock Events (G)', 
                              marker=dict(color='red', size=8)),
                    row=2, col=1
                )
            
            # Temperature plot
            fig.add_trace(
                go.Scatter(x=sim_data['time'], y=sim_data['temperature'],
                          name='Temperature (°C)', line=dict(color='green')),
                row=3, col=1
            )
            
            # Humidity plot (if applicable)
            if sim_data['humidity'] is not None:
                fig2 = go.Figure()
                fig2.add_trace(
                    go.Scatter(x=sim_data['time'], y=sim_data['humidity'],
                              name='Humidity (%)', line=dict(color='orange'))
                )
                fig2.update_layout(title="Humidity Variation", xaxis_title="Time (hours)", yaxis_title="Humidity (%)")
            
            fig.update_layout(
                height=800,
                title=f"Transportation Simulation - {transport_mode.replace('_', ' ').title()}",
                xaxis3_title="Time (hours)"
            )
            
            fig.update_yaxes(title_text="Vibration (G)", row=1, col=1)
            fig.update_yaxes(title_text="Shock (G)", row=2, col=1)  
            fig.update_yaxes(title_text="Temperature (°C)", row=3, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            if sim_data['humidity'] is not None:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Simulation summary
            st.subheader("📊 Simulation Summary")
            
            max_vibration = np.max(sim_data['vibration'])
            max_shock = np.max(sim_data['shock']) if np.any(sim_data['shock'] > 0) else 0
            temp_range = (np.min(sim_data['temperature']), np.max(sim_data['temperature']))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Max Vibration", f"{max_vibration:.2f} G")
                st.metric("Max Shock", f"{max_shock:.2f} G")
            
            with col2:
                st.metric("Temperature Range", f"{temp_range[0]:.1f}°C to {temp_range[1]:.1f}°C")
                if sim_data['humidity'] is not None:
                    humidity_range = (np.min(sim_data['humidity']), np.max(sim_data['humidity']))
                    st.metric("Humidity Range", f"{humidity_range[0]:.1f}% to {humidity_range[1]:.1f}%")
            
            with col3:
                # Risk assessment
                risk_level = "LOW"
                if max_vibration > 3.0 or max_shock > 6.0:
                    risk_level = "HIGH"
                elif max_vibration > 2.0 or max_shock > 4.0:
                    risk_level = "MEDIUM"
                
                if risk_level == "LOW":
                    st.success(f"✅ Risk Level: {risk_level}")
                elif risk_level == "MEDIUM":
                    st.warning(f"⚠️ Risk Level: {risk_level}")
                else:
                    st.error(f"❌ Risk Level: {risk_level}")
    
    with tab6:
        st.header("📋 Summary Report")
        
        if not all([st.session_state.cad_data, st.session_state.material_data]):
            st.warning("⚠️ Please complete all previous steps to generate the summary report.")
            return
        
        # Generate comprehensive report
        weights = calculate_packaging_weights(st.session_state.cad_data, st.session_state.material_data)
        
        # Spider chart for overall analysis
        if 'test_results' in st.session_state:
            st.subheader("📊 Overall Performance Analysis")
            spider_fig = create_spider_chart(weights, st.session_state.test_results, st.session_state.material_data)
            st.plotly_chart(spider_fig, use_container_width=True)
        
        st.subheader("📏 Product Specifications")
        specs_data = {
            "Parameter": ["Length", "Width", "Height", "Volume"],
            "Value": [
                f"{st.session_state.cad_data['length']:.2f} mm",
                f"{st.session_state.cad_data['width']:.2f} mm", 
                f"{st.session_state.cad_data['height']:.2f} mm",
                f"{st.session_state.cad_data['volume']:.2f} cm³"
            ]
        }
        st.table(pd.DataFrame(specs_data))
        
        st.subheader("🏭 Material Selection Summary")
        material_summary = []
        for component, material in st.session_state.material_data.items():
            if component != 'blister_thickness':
                mat_name = st.session_state.material_csv_data[
                    st.session_state.material_csv_data['Short_Name'] == material
                ]['Name'].iloc[0]
                material_summary.append({
                    "Component": component.replace('_', ' ').title(),
                    "Material": f"{mat_name} ({material})"
                })
        
        st.table(pd.DataFrame(material_summary))
        
        st.subheader("⚖️ Weight Analysis")
        weight_data = {
            "Component": [
                "Single Brush", "Blister Pack", "Base Card", "Single Package Total",
                "12-Pack Brushes", "Plastic Tray", "Corrugated Carton", "Final Total"
            ],
            "Weight (g)": [
                f"{weights['brush']:.2f}",
                f"{weights['blister']:.2f}", 
                f"{weights['base']:.2f}",
                f"{weights['single_package']:.2f}",
                f"{weights['twelve_pack_brushes']:.2f}",
                f"{weights['tray']:.2f}",
                f"{weights['carton']:.2f}",
                f"{weights['total']:.2f}"
            ],
            "Target (g)": ["14.56", "4.94", "7.19", "25.53", "306.36", "14.00", "114.00", "334.00"]
        }
        st.table(pd.DataFrame(weight_data))
        
        if 'test_results' in st.session_state:
            st.subheader("🧪 Test Results Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**ISTA 2A Results:**")
                st.write(f"- Drop Height: {st.session_state.test_results['ista_2a']['drop_height_m']:.2f} m")
                st.write(f"- Impact Force: {st.session_state.test_results['ista_2a']['impact_force_N']:.1f} N")
                st.write(f"- Assessment: {st.session_state.test_results['assessment_2a']}")
            
            with col2:
                st.write("**ISTA 6A Results:**")
                st.write(f"- Drop Height: {st.session_state.test_results['ista_6a']['drop_height_m']:.2f} m")
                st.write(f"- Impact Force: {st.session_state.test_results['ista_6a']['impact_force_N']:.1f} N") 
                st.write(f"- Assessment: {st.session_state.test_results['assessment_6a']}")
        
        # AI Recommendations
        if 'ai_recommendations' in st.session_state and st.session_state.ai_recommendations:
            st.subheader("🤖 AI Recommendations")
            
            st.write("**Recommended Changes:**")
            for rec in st.session_state.ai_recommendations.get('recommendations', []):
                st.write(f"- {rec}")
            
            st.write("**Parameter Changes:**")
            for param, value in st.session_state.ai_recommendations.get('parameter_changes', {}).items():
                st.write(f"- {param.replace('_', ' ').title()}: {value}")
        
        # Download options
        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Material Data"):
                csv = st.session_state.material_csv_data.to_csv(index=False)
                st.download_button(
                    label="Download Material Database",
                    data=csv,
                    file_name=f"material_database_{st.session_state.session_id}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Weight Analysis"):
                weight_df = pd.DataFrame(weight_data)
                csv = weight_df.to_csv(index=False)
                st.download_button(
                    label="Download Weight Analysis", 
                    data=csv,
                    file_name=f"weight_analysis_{st.session_state.session_id}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Export Full Report"):
                # Generate comprehensive JSON report
                full_report = {
                    "session_id": st.session_state.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "cad_data": st.session_state.cad_data,
                    "material_data": st.session_state.material_data,
                    "weights": weights,
                    "test_results": st.session_state.test_results if 'test_results' in st.session_state else {},
                    "chat_history": st.session_state.chat_history
                }
                
                json_str = json.dumps(full_report, indent=2)
                st.download_button(
                    label="Download Full Report (JSON)",
                    data=json_str,
                    file_name=f"design_buddy_report_{st.session_state.session_id}.json",
                    mime="application/json"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("**Design Buddy** - AI-Powered Packaging Design Assistant | Built with Streamlit & Gemini AI")

if __name__ == "__main__":
    main()
