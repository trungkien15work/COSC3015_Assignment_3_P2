#!/bin/bash
# Kill any existing processes
pkill -f gunicorn

cd /home/ubuntu/my-flask-app

# Start the Flask app with Gunicorn
nohup gunicorn --bind 0.0.0.0:5000 application:application &