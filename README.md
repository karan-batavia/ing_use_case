# ING Use Case - Prompt Scrubber API

A comprehensive text processing and audit system for sensitive data redaction, AI-powered predictions, and secure de-scrubbing capabilities.

**For the ML model, look at the ML_setup_01 branch.**

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.12+ (for local development)
- Gemini API key (for AI predictions)

### 1. Environment Setup

Create a `.env` file in the project root:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Start the Services

```bash
# Using Docker Compose
docker-compose up -d

# Or using the provided script
./docker-start.sh start
```

This will start:
- **MongoDB** (port 27017) - Database for audit logs
- **FastAPI** (port 8000) - REST API backend  
- **Streamlit** (port 8501) - Web interface

### 3. Access the Applications

- **Streamlit Web App**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 🌱 Database Seeding

To explore the audit logs and test the de-scrubbing functionality, you can seed the database with realistic test data.

### Seed with Sample Data

The `seed_database.py` script creates realistic audit log entries based on actual prompt examples:

```bash
# Make sure containers are running first
docker-compose up -d

# Run the seeding script
python seed_database.py
```

### Interactive Options

When you run the script, you'll see:

```
🌱 ING Use Case - Database Seeder
This will create realistic audit log entries for testing and exploration.

Options:
1. Seed database (keep existing data)
2. Clear existing data and seed fresh  
3. Just clear existing data

Enter your choice (1-3):
```

**Recommended**: Choose option **2** for a clean start with fresh test data.

### What Gets Created

The seeder generates **25 realistic workflow sessions** with:
- 📝 **Text redactions** (100% of sessions) - Original prompts with sensitive data redacted
- 🤖 **AI predictions** (70% of sessions) - AI-generated responses using redacted prompts
- 🔓 **De-scrubbing operations** (30% of sessions, admin only) - Restored original text from redacted versions

### Sample Data Types

Based on real ING prompt examples:
- Financial reports and disclosures
- Customer agreement processing
- Infrastructure capacity planning  
- Incident response procedures
- Compliance documentation
- Employee contact information

### Example Output

```
🎉 Database seeding completed!
   📊 Total sessions created: 25
   👥 Users involved: 5
   📝 Sample prompt types: 6

📈 Seeding Summary:
   Total interactions: 67
   Unique sessions: 25
   Redactions: 25
   Predictions: 18
   De-scrubs: 7
```

## 🔧 Development

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start only MongoDB
docker-compose up mongodb -d

# Run FastAPI locally
uvicorn api:app --reload --port 8000

# Run Streamlit locally  
streamlit run streamlit_app.py --server.port 8501
```

### API Endpoints

- `POST /redact` - Redact sensitive information from text
- `POST /predict` - Generate AI predictions using redacted text
- `POST /de-scrub` - Restore original text (admin only)
- `GET /redaction-records` - View audit logs (admin only)

### Authentication

The API uses JWT authentication. Admin-level access is required for:
- De-scrubbing operations (`/de-scrub`)
- Viewing audit logs (`/redaction-records`)

## 📊 Data Flow

```
Original Text → [Redact] → Redacted Text → [Predict] → AI Response
                    ↓                           ↓
              Audit Log Entry            Audit Log Entry
                    ↓
            [De-scrub] ← Admin Access Required
                    ↓
            Original Text Restored
```

## 🛠️ Architecture

- **FastAPI**: REST API with JWT authentication and admin role management
- **MongoDB**: Session tracking, audit logs, and redaction record storage
- **Streamlit**: User-friendly web interface for text processing
- **Presidio**: Microsoft's data loss prevention library for entity detection
- **Gemini AI**: Google's AI service for text generation
- **scikit-learn**: Machine learning pipeline for sensitivity classification

## 📁 Project Structure

```
ing_use_case/
├── api.py                 # FastAPI application
├── streamlit_app.py       # Streamlit web interface
├── seed_database.py       # Database seeding script
├── docker-compose.yml     # Container orchestration
├── requirements.txt       # Python dependencies
├── src/
│   ├── mongodb_service.py # Database operations
│   ├── presidio.py        # Text redaction service
│   └── prompt_scrubber.py # Sensitivity classification
└── data/
    ├── classification/    # Training data for ML models
    └── prompts/          # Example prompts for seeding
```

## 🔍 Exploring Audit Logs

After seeding the database, you can explore the audit trails:

1. **Via Streamlit**: Access the web interface at http://localhost:8501
2. **Via API**: Use the FastAPI docs at http://localhost:8000/docs
3. **Direct MongoDB**: Connect to `mongodb://localhost:27017` with credentials from docker-compose.yml

## 🚦 Troubleshooting

### Container Issues
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs fastapi-app
docker-compose logs mongodb

# Restart services
docker-compose restart
```

### Database Connection Issues
```bash
# Test MongoDB connection
docker-compose exec mongodb mongosh -u inguser -p ingpassword --authenticationDatabase admin
```

### Seeding Issues
```bash
# Check if MongoDB is accessible
python -c "from src.mongodb_service import MongoDBService; print('Connected:', MongoDBService().is_connected())"
```
