import sys

sys.path.insert(0, './models')
sys.path.insert(1, './data_preprocess')
sys.path.insert(2, './UI')
sys.path.insert(3, './models/mlruns')

from app import QTApp

if __name__ == '__main__':
    app = QTApp()
    app.run()
