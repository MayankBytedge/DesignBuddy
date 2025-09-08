# 🤖 Design Buddy - AI-Powered Packaging Design Assistant

Design Buddy is an advanced conversational AI agent specialized in toothbrush packaging design, testing, and optimization. It combines CAD analysis, material science, ISTA testing standards, and real-time AI guidance to help engineers design optimal packaging solutions.

## 🚀 Features

### Core Capabilities
- **CAD File Analysis**: Upload .stp, .x_t, .iges files and extract precise dimensions using CadQuery
- **Conversational AI**: Powered by Google Gemini 2.0 Flash for intelligent guidance
- **Material Database**: Comprehensive database of packaging materials with properties
- **Weight Calculations**: Accurate weight calculations based on volume × density
- **ISTA Testing**: Complete ISTA 2A and ISTA 6A testing simulations
- **Transportation Simulation**: Real-world shipping simulation including sea, air, ground, rail
- **AI Optimization**: Parameter optimization and design alternatives
- **Workshop Mode**: Interactive parameter adjustment with real-time results

### Specialized for Toothbrush Packaging
- Blister pack design (0.2-0.5mm thickness range)
- Cardboard backing sheets
- 12-unit tray configurations
- Corrugated carton optimization
- Complete weight validation (targeting ~334g total)

### Testing & Analysis
- **ISTA 2A**: Non-simulation integrity testing
- **ISTA 6A**: Enhanced simulation for distribution
- Drop test calculations with impact forces
- Compression testing with load factors
- Vibration analysis with frequency sweeps
- Material failure risk assessment
- Transportation environmental factors

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd design-buddy
   ```

2. **Create virtual environment**
   ```bash
   python -m venv design_buddy_env
   
   # On Windows
   design_buddy_env\Scripts\activate
   
   # On macOS/Linux
   source design_buddy_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run design_buddy_app.py
   ```

## 🔧 Configuration

### Environment Variables
Create a `.env` file with the following variables:

```env
# Required
GEMINI_API_KEY=your_google_gemini_api_key

# Optional
DEBUG_MODE=False
DEFAULT_MATERIAL=PP
DEFAULT_THICKNESS=0.3
```

### Getting Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## 🎯 Usage Guide

### Step 1: CAD Upload
- Upload your toothbrush CAD file (.stp, .x_t, .iges)
- Design Buddy extracts dimensions automatically
- Review extracted length, width, height, and volume

### Step 2: Material Selection
- Choose materials for each component:
  - Toothbrush handle (HDPE, PP, PET)
  - Blister pack (PET, PP)
  - Bottom sheet (Cardboard, Solid Board)
  - Tray (PP, HDPE)
  - Outer carton (Corrugated options)

### Step 3: Weight Analysis
- Automatic weight calculations
- AI validation of realistic weights
- Comparison with industry targets

### Step 4: Testing Configuration
- Select ISTA test types (2A, 6A)
- Choose transportation mode
- Set weather conditions
- Configure distance parameters

### Step 5: Results & Analysis
- **Summary Tab**: Overview of design performance
- **ISTA Testing Tab**: Detailed test results and pass/fail analysis
- **Transportation Tab**: Real-world shipping simulation
- **Analysis Tab**: AI insights and spider charts
- **Workshop Tab**: Interactive parameter optimization

## 📊 Material Database

The app includes a comprehensive material database with properties for:

### Plastics
- **HDPE**: High-density polyethylene for handles
- **PP**: Polypropylene for flexible components
- **PET**: Clear blister materials

### Cardboard
- **3-Ply Corrugated**: Standard shipping boxes
- **5-Ply Corrugated**: Heavy-duty applications
- **E-Flute**: Thin profile corrugated
- **Solid Fiber Board**: Premium backing

Each material includes:
- Density (g/cm³)
- Young's Modulus (MPa)
- Poisson Ratio
- Yield/Ultimate Strength (MPa)
- Cost per kg
- Temperature ranges
- Recyclability data

## 🧪 ISTA Testing Standards

### ISTA 2A - Non-Simulation Tests
- Drop heights based on package weight
- 2x compression load factor
- 1-200 Hz vibration testing
- 24-hour duration

### ISTA 6A - Enhanced Simulation
- Amazon-specific parameters
- 1.8x compression load factor
- Extended 5-500 Hz vibration
- 72-hour extended testing
- Climate conditioning

## 🚛 Transportation Simulation

Realistic simulations for:
- **Ground**: Truck transport with traffic patterns
- **Air**: Cargo hold conditions and turbulence
- **Sea**: Wave effects and extended duration
- **Rail**: Coupling shocks and long distances

Environmental factors:
- Temperature variations
- Humidity levels
- Weather conditions
- Route-specific challenges

## 🤖 AI Integration

Design Buddy uses Google Gemini 2.0 Flash for:
- Conversational guidance
- Material selection advice
- Test result interpretation
- Design optimization suggestions
- Failure analysis
- Alternative design generation

## 📈 Advanced Features

### Workshop Mode
- Real-time parameter adjustment
- What-if analysis
- Material substitution
- Thickness optimization
- Cost-performance trade-offs

### AI Optimization
- Automatic parameter tuning
- Multi-objective optimization
- Cost reduction strategies
- Performance enhancement
- Sustainability improvements

### Reporting
- Comprehensive test reports
- Spider chart analysis
- Weight breakdown charts
- Transportation performance graphs
- Material comparison tables

## 🔍 Technical Details

### CAD Processing
- Uses CadQuery for STEP/IGES file processing
- Automatic bounding box calculation
- Volume extraction for weight analysis
- Support for complex geometries

### Force Calculations
- Drop impact using energy conservation
- Compression stress analysis
- Vibration dynamic loading
- Edge crush strength requirements
- Material failure predictions

### Material Science
- Stress-strain relationships
- Safety factor calculations
- Temperature effects
- Humidity impacts
- Long-term durability

## 🐛 Troubleshooting

### Common Issues

**CAD File Not Loading**
- Ensure file format is supported (.stp, .x_t, .iges)
- Check file size (< 50MB recommended)
- Verify file integrity

**Gemini API Errors**
- Verify API key in .env file
- Check internet connection
- Ensure API quota is available

**Import Errors**
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Verify virtual environment activation

### Debug Mode
Enable debug mode in .env:
```
DEBUG_MODE=True
```

## 📄 License

Design Buddy – All Rights Reserved – BytEdge

## 🤝 Contributing

This is a proprietary application. For feature requests or bug reports, please contact the development team.

## 📞 Support

For technical support or questions:
- Email: support@byteedge.com
- Documentation: [Internal Wiki]
- Issue Tracker: [Internal System]

---

**Design Buddy v2.0** - Revolutionizing packaging design with AI-powered analysis and optimization.