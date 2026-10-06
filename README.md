# NEXUS Analytics Studio

NEXUS Analytics Studio is a full-stack analytics platform designed to transform uploaded datasets into interactive dashboards, domain-specific analytics, cross-file comparisons, AI-assisted insights, and executive reports.

## Key Features

- Multi-format dataset ingestion
- Automated data profiling and quality analysis
- Interactive analytics dashboards
- Multi-file comparison
- AI Analyst for dataset-driven insights
- Executive intelligence and reporting
- Dynamic charts and KPI visualization
- FastAPI backend
- Next.js frontend

## Analytics Modules

NEXUS currently supports 12 analytics domains:

1. Sales Analytics
2. Customer Analytics
3. Marketing Analytics
4. Financial Analytics
5. Supply Chain Analytics
6. Product Analytics
7. Operations Analytics
8. HR Analytics
9. Fraud Analytics
10. Healthcare Analytics
11. Manufacturing Analytics
12. E-commerce Analytics

## Architecture

### Frontend

- Next.js
- TypeScript
- Tailwind CSS
- Recharts

### Backend

- Python
- FastAPI
- Pandas
- Domain-specific analytics engines

## Local Development

### Backend

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r .\api\requirements-api.txt

Start the FastAPI backend:
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

API documentation:
http://127.0.0.1:8000/docs
Frontend
Open another terminal:
cd frontend
npm install
npm run dev

Open the application:
http://localhost:3000
Core Workflow
Dataset Upload → Data Profiling → Domain Analytics → Visualization → AI Analyst → Executive Reporting
Project Status
NEXUS Analytics Studio v1.0
- 12 analytics modules
- Multi-file analytics
- Interactive visualization
- AI-assisted analysis
- Executive reporting
- Full-stack frontend/backend integration