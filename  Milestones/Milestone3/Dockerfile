FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml --quiet && \
    conda clean --all -f -y

ENV PATH /opt/conda/envs/myenv/bin:$PATH

COPY model.data .
COPY main.py .

CMD ["python", "main.py"]