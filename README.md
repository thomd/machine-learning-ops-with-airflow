# Machine Learning Training using Apache Airflow

This is an **educational project**

## Setup

    pyenv shell 3.10.9
    python -m venv .venv
    source .venv/bin/activate
    export AIRFLOW_HOME=$(pwd)
    export SQLALCHEMY_SILENCE_UBER_WARNING=1
    export AIRFLOW__CORE__LOAD_EXAMPLES=False
    pip install apache-airflow
    airflow db init
