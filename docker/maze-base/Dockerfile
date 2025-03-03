FROM python:3.11.11-bookworm

RUN pip install uv
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --system
