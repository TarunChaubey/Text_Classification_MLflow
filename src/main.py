import mlflow
import os
if __name__ == '__main__':
    with mlflow.start_run() as run:
        mlflow.run('src/',"get_prepare_data.py")
        mlflow.run('src/',"train.py")
        mlflow.run('src/',"evaluate.py")
        # os.system("python src\get_prepare_data.py")
        # mlflow.run(".", "get_prepare_data", use_conda=False)
        # mlflow.run(".", "train", use_conda=False)
        # mlflow.run(".", "evaluate", use_conda=False)