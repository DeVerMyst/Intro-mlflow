**venv**
`conda create -n my_env python=3.8 scikit-learn mlflow`
`conda activate my_env`
**export**
`conda env export --no-builds > my_env.yml`

**api**
`uvicorn mlflow_api:app --reload`

`curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"data": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]}'`

>{"prediction":[10910.802840896466]}