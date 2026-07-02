# Setup
Windows: <br />
run setup.bat <br />
run main.py with python <br />

Linux:
```bash
cd frontend
npm install
npm run build
cd ..
python -m pip install requirements.txt
python main.py
```

This should all run a server at localhost:8000, otherwise the port will be indicated in the startup message.

# Notes
This repository is a combination of two children, fernando_input_gui and fernando_output_gui, If you are receiveing this repository to work on it is assumed that you have these two sub repositories. 

The two repositories were combined by putting all python backend functions into one file, writing a simple landing page that switches between the apps. The directories frontend/src/{input | output} contain the files directly from the frontend/src directories of the two children minus the main.jsx files.

More details about the spesific implimentations of each app and some of the decisions made are documented in the README's of the two sub repositories, for convenience, those can also be fount in the readmes directory of this repository.
