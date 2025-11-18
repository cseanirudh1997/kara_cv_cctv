# 🏢 CCTV Monitoring System

An advanced region-wise CCTV monitoring system with multilingual AI chatbot support.

## 🚀 Features

### 📊 Region-wise Monitoring
- **🍳 Kitchen Region**: Personnel tracking, temperature monitoring, fire safety alerts
- **🏛️ Hall Region**: Visitor counting, crowd density analysis, table occupancy tracking  
- **🚗 Parking Region**: Vehicle detection, entry/exit logging, parking violations

### 🤖 Multilingual AI Chatbot
- **English**: Natural language queries about system status
- **Hindi**: देसी भाषा में सिस्टम की जानकारी
- **Hinglish**: Mixed language support for easy communication

### 🎥 Advanced Analytics
- Real-time camera feed simulation
- Historical trend analysis
- Alert distribution and priority management
- Performance metrics and system health monitoring

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **Web Interface**: Gradio 4.0+
- **Database**: SQLite3
- **Visualization**: Plotly, Matplotlib
- **Data Processing**: Pandas, NumPy
- **NLP**: NLTK, TextBlob

## 📦 Installation

1. **Clone or download the project**:
   ```bash
   cd MPT-Karan-5
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python cctv_monitoring_system.py
   ```

4. **Access the dashboard**:
   Open your browser and go to: `http://localhost:7860`

## 🎯 Usage

### 🏠 Control Center
- View system-wide overview
- Monitor all regions simultaneously  
- Check camera status and system health
- Access emergency controls

### 📹 Region Monitoring
Each region has dedicated monitoring with:
- Live camera feed simulation
- Event logging and tracking
- Performance analytics
- Quick action controls

### 💬 AI Assistant
Ask questions in natural language:

**English Examples:**
- "How many cameras are online in the kitchen?"
- "Show me recent parking violations"
- "What's the temperature in the kitchen?"

**Hindi Examples:**
- "Kitchen mein kitne cameras chalu hain?"
- "Parking area mein kya problem hai?"
- "Hall mein kitne log hain?"

**Hinglish Examples:**
- "Kitchen cameras ka status kya hai?"
- "Parking mein vehicles ki count batao"
- "System health kaisa hai?"

## 📊 Database Schema

### Tables Created:
1. **kitchen_events**: Kitchen-specific monitoring data
2. **hall_events**: Hall visitor and activity tracking
3. **parking_events**: Vehicle and parking management
4. **camera_status**: Camera health and performance
5. **system_alerts**: System-wide alerts and notifications

## 🔧 Configuration

### Camera Setup:
- **Kitchen**: 3 cameras (KIT-CAM-01, KIT-CAM-02, KIT-CAM-03)
- **Hall**: 2 cameras (HALL-CAM-01, HALL-CAM-02)  
- **Parking**: 4 cameras (PARK-CAM-01 to PARK-CAM-04)

### Supported Alert Types:
- Fire and smoke detection
- Unauthorized access
- Equipment malfunctions
- Crowd management
- Vehicle violations
- System performance issues

## 📈 Sample Data

The system comes pre-loaded with realistic sample data including:
- 48 hours of event history per region
- Camera performance metrics
- Alert distribution across different priorities
- System health logs

## 🚀 Advanced Features

### Real-time Monitoring
- Live camera status updates
- Auto-refresh capabilities
- Performance metrics tracking
- Alert priority management

### Analytics Dashboard
- Historical trend analysis
- Performance benchmarking
- Resource utilization monitoring
- Comprehensive reporting

### System Health
- Component status monitoring
- Performance metrics tracking
- Error rate analysis
- Uptime monitoring

## 🛡️ Security Features

- Region-based access control
- Event logging and audit trails
- Emergency lockdown capabilities
- Alert escalation system

## 🔄 Auto-refresh

The dashboard supports auto-refresh functionality to keep data current:
- 30-second refresh intervals
- Real-time camera status updates
- Live event streaming simulation
- Performance metrics updates

## 📱 Mobile Responsive

The Gradio interface is mobile-responsive and works on:
- Desktop computers
- Tablets
- Mobile phones
- Large displays

## 🎨 Customization

### Adding New Regions:
1. Update the database schema
2. Create new region monitor class
3. Add to the main system controller
4. Update the Gradio interface

### Language Support:
Add new language patterns in the `MultilingualChatbot` class query patterns dictionary.

## 🐛 Troubleshooting

### Common Issues:

**Port Already in Use:**
```bash
# Kill process using port 7860
lsof -ti:7860 | xargs kill -9
```

**Database Issues:**
- Delete `cctv_monitoring.db` to reset database
- Check file permissions
- Ensure SQLite3 is installed

**Package Installation:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📊 Performance

### System Requirements:
- **RAM**: Minimum 2GB, Recommended 4GB
- **CPU**: Multi-core processor recommended
- **Storage**: 500MB for application + database growth
- **Network**: Local network access for remote monitoring

### Expected Performance:
- **Response Time**: <200ms for most queries
- **Concurrent Users**: 10-50 depending on hardware
- **Database Operations**: 1000+ ops/second
- **Camera Simulation**: 15-30 FPS per camera

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is for educational and demonstration purposes.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review the code comments
- Test with sample data first
- Verify all dependencies are installed

---

**Built with ❤️ for comprehensive CCTV monitoring**