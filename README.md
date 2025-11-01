# CTU ROB Semestral Project


## Installation
```bash
python -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```
## Web streaming (MJPEG)

This project now includes a simple integrated web server that streams the robot camera as an MJPEG feed.

- Run the server:

```bash
python3 stream_server.py
```

Then open http://localhost:5000 in your browser. The server uses Flask and OpenCV; install dependencies with:

```bash
pip install -r requirements-server.txt
```

The server will use the `ctu_crs.CRS97` (or `CRS93`) robot interface if available; if not, a placeholder (black) frame is served so the UI still works for development.
