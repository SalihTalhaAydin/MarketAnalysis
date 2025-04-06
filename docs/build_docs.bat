@echo off
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

echo Building Sphinx documentation...
sphinx-build -b html source _build/html

echo Deactivating virtual environment...
deactivate

echo Documentation build complete. Check the _build/html directory.
pause
