# NEXUS API

This API runs inside the existing `nexus-analytics-studio` Python project.

## Run

From:

`C:\Users\DELL\Downloads\nexus-analytics-studio`

install:

```powershell
pip install -r .\api\requirements-api.txt
```

run:

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Health:

`http://127.0.0.1:8000/api/health`

Docs:

`http://127.0.0.1:8000/docs`
