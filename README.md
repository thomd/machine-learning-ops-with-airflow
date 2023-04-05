# Machine Learning Training using Apache Airflow

This is an **educational project**

## Pipeline

```
airflow dags show ml | sed 1d | graph-easy --as=boxart

╭──────────────────╮     ╭───────────────────────╮     ╭────────────────────╮     ╭─────────────────────╮
│ download_dataset │ ──▶ │    data_processing    │ ──▶ │ training_logisitic │ ──▶ │ identify_best_model │
╰──────────────────╯     ╰───────────────────────╯     ╰────────────────────╯     ╰─────────────────────╯
                           │                                                        ▲
                           │                                                        │
                           ▼                                                        │
                         ╭───────────────────────╮                                  │
                         │ training_randomforest │ ─────────────────────────────────┘
                         ╰───────────────────────╯
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
