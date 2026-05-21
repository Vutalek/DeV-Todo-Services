# syntax=docker/dockerfile:1

# Comments are provided throughout this file to help you get started.
# If you need more help, visit the Dockerfile reference guide at
# https://docs.docker.com/go/dockerfile-reference/

# Want to help us make this template better? Share your feedback here: https://forms.gle/ybq9Krt8jtBL3iCk7

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to requirements.txt to avoid having to copy them into
# into this layer.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Create directories for RAG persistence and immutable seed data.
RUN mkdir -p /app/db/chroma_db /app/seed && chown -R appuser:appuser /app/db /app/seed

# Copy the source code into the container.
COPY --chown=appuser:appuser . .

# Run Apache issues parser to populate initial seed data when Jira is reachable.
# The image must still build if Jira is slow/down; the app can start with an
# empty Chroma collection and seed later from a newer image.
RUN APACHE_ISSUES_CSV_PATH=/app/seed/apache_issues.csv python db/parse_apache_issues.py || echo "Warning: failed to fetch Jira seed data; continuing without seed CSV"
RUN chown -R appuser:appuser /app/db /app/seed

VOLUME ["/app/db/chroma_db"]

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application. Keep one worker because BM25 lives in process memory.
CMD uvicorn app.app:app --host 0.0.0.0 --port 8000 --workers 1
