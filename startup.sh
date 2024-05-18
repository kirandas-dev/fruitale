# startup.sh

#!/bin/bash

# Install necessary system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx

# Activate the virtual environment
source antenv/bin/activate

# Start the application
python3 app.py
