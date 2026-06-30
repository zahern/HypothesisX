@echo off
cd frontend
call npm install
call npm run build
cd ..
python -m pip install -r requirements.txt
