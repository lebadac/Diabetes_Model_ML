FROM python:3.10
LABEL authors = 'Dac'

WORKDIR /src

COPY diabetes_model_logisticregression.joblib /src/diabetes_model_logisticregression.joblib
COPY server.py /src/server.py
COPY requirements_fordocker.txt /src/requirements_fordocker.txt

RUN pip install -r requirements_fordocker.txt

EXPOSE 8888

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8888"]