#!/bin/bash
vlc --extraintf=http --http-host 127.0.0.1 --http-port 8091 &
python vlc_controller.py

