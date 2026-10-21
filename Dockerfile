FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_X11_NO_MITSHM=1

# opencv-python's Qt window needs OpenGL, glib, and the X11 client libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libx11-6 \
        libxcb1 \
        libxkbcommon0 \
        libxkbcommon-x11-0 \
        libdbus-1-3 \
        libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Live discovery needs OpenCV and NumPy only. SciPy and Matplotlib stay in
# requirements.txt for the offline scripts. data/ (video samples) is not copied.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        $(grep -E '^(opencv-python|numpy)>=' requirements.txt)

COPY ball_tracker.py calibrate.py physics_discovery.py live.py ./

CMD ["python", "live.py"]
