# Getting Started

Mit dieser Anleitung kann man den Skript zur Analyse der Weiterverrechnungsdateien starten.

## Prerequisites
This is the list of things you need to have and how to install them.

Open Anaconda Prompt and follow steps 1-5 if the program is executed for the first time. Otherwise,
skip to the Execution part.

##### 1. Change directory to folder with extracted files (directory_path) 
```
cd "directory_path"
```
##### 2. Install virtualenv
```
pip install virtualenv
```
##### 3. Create virtual environment
```
virtualenv venv --python=python3.9
```
##### 4. Activate virtual env
```
.\venv\Scripts\activate
```
##### 5. Install libraries
```
pip install -r requirements.txt
```

### Execution
Execution of the script in `powershell` as follows:
```
streamlit run main.py
```
