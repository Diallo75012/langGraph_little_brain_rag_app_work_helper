#!/bin/bash

# This script will create the cronjob to clear the cache, we can update it to change the periodicity of cache clearance


# Path to the Python executable
PYTHON_PATH="/usr/bin/python3"

# Path to the clear_cache.py script
SCRIPT_PATH="/home/creditizens/langgraph/clear_cache.py"

# Cron job entry
CRON_JOB="0 0 * * * $PYTHON_PATH $SCRIPT_PATH"

CURRENT_CRON=$(crontab -l 2>/dev/null)

# Check if the cron job already exists
if echo "$CURRENT_CRON" | grep -q "$SCRIPT_PATH"; then
    echo "Cron job already exists."
else
    (echo "$CURRENT_CRON"; echo "$CRON_JOB") | crontab -
    echo "Cron job set up to run clear_cache.py daily at midnight."
fi
