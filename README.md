# Machine Learning Training using Apache Airflow

This is an **educational project** and **proof-of-concept**.

The pipeline trains a **Random Forest classifier** and a **Logistic Regression classifier** using the 
[Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) and identifies the best model by accuracy.

## ML Pipeline

```
airflow dags show ml | sed 1d | graph-easy --as=boxart

╭──────────────────╮     ╭────────────────────╮     ╭────────────────╮     ╭─────────────────────╮
│ download_dataset │ ──▶ │  data_processing   │ ──▶ │ train_logistic │ ──▶ │ identify_best_model │
╰──────────────────╯     ╰────────────────────╯     ╰────────────────╯     ╰─────────────────────╯
                           │                                                 ▲
                           │                                                 │
                           ▼                                                 │
                         ╭────────────────────╮                              │
                         │ train_randomforest │ ─────────────────────────────┘
                         ╰────────────────────╯
```

## Setup

    pyenv shell 3.10.9
    python -m venv .venv
    source .venv/bin/activate
    export AIRFLOW_HOME=$(pwd)
    export SQLALCHEMY_SILENCE_UBER_WARNING=1
    export AIRFLOW__CORE__LOAD_EXAMPLES=False
    pip install apache-airflow numpy pandas scikit-learn
    airflow db init
    airflow scheduler

## Train

    airflow dags unpause ml
    airflow dags trigger ml
    cat accuracy.txt
