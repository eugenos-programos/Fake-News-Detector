import mlflow


if __name__ == '__main__':
    mlflow.projects.run(uri='../fake-news-detector', docker_args='', build_image=True)
