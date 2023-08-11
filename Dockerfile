FROM python:3.8

# Install system packages
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    ffmpeg

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /src

COPY ./prompts /prompts

COPY ./app /app

WORKDIR /app

EXPOSE 8080

CMD ["python", "app.py"]
