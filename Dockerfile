# Use full Debian base to install build tools
FROM python:3.11

# Install OS dependencies required to build dlib
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip and install requirements
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Start your app (replace app:app with actual entry point)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
