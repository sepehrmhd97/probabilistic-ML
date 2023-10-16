# Running the Project

Follow the instructions below to set up and run the project:

## 1. Setup

To set up the necessary environment and dependencies, execute the following command:

\```
make
\```

This command will:

- Check if a virtual environment named `APMLvenv` exists.
- If it doesn't exist, it will create the virtual environment.
- Install the necessary dependencies from `requirements.txt` into the virtual environment.

## 2. Activate the Virtual Environment

Before running the `runme.py` script, you need to activate the virtual environment. To do this, execute the following command:

\```
source APMLvenv/bin/activate
\```

You should see the command prompt change, indicating that you are now inside the virtual environment.

## 3. Run the Script

With the virtual environment activated, run the `runme.py` script with:

\```
python runme.py
\```

## 4. Exiting the Virtual Environment

After you've run the script and want to exit the virtual environment, simply execute:

\```
deactivate
\```

This will bring you back to your system's default Python environment.

---
