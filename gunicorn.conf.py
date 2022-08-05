# -*- coding: utf-8 -*-
"""
Gunicorn with Uvicorn config to launch in Digital Ocean's App Platform.
"""
bind = "0.0.0.0:8080"
workers = 4
client_header_timeout = 3000
client_body_timeout = 3000
fastcgi_read_timeout = 3000
timeout = 120
threads = 16
preload = True
# Uvicorn's Gunicorn worker class
worker_class = "uvicorn.workers.UvicornWorker"