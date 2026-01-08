# Development environment helper scripts

## Windows (PowerShell)

### Activate venv and run server:
```powershell
.\run.ps1 uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Install new package:
```powershell
.\run.ps1 pip install <package-name>
```

### Run tests:
```powershell
.\run.ps1 pytest tests/
```

## Linux/Mac (Bash)

### Activate venv and run server:
```bash
./run.sh uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

### Install new package:
```bash
./run.sh pip install <package-name>
```

### Run tests:
```bash
./run.sh pytest tests/
```

## Manual Activation

### Windows:
```powershell
.\venv\Scripts\Activate.ps1
```

### Linux/Mac:
```bash
source venv/bin/activate
```

## Initial Setup

1. Create virtual environment (if not exists):
   ```bash
   python -m venv venv
   ```

2. Activate it (see above)

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080 --reload
   ```
