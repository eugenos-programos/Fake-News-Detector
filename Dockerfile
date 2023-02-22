FROM python:3.10

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY build_project.py run_project.py .

CMD ["python3", "build_project.py"]
