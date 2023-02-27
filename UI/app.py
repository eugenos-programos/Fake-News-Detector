from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QGridLayout, QWidget, QComboBox, QMessageBox, QErrorMessage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QSize, Qt
import os
from controller import Controller
import time


class MainWindow(QMainWindow):
    def __init__(self, app) -> None:
        super().__init__()

        self.app = app
        self.setWindowTitle("Fake news detector")
        self.setFixedSize(QSize(1100, 1000))
        self.create_layout()
    
    def create_button(self):
        self.button = QPushButton("Predict fakeness")
        self.button.clicked.connect(self.click_button)
        self.button.setFixedSize(QSize(120, 40))

    def create_label_info(self):
        self.label = QLabel("Fake news detector. Type url for predicting.")
        self.label_font = self.label.font()
        self.label_font.setPointSize(24)
        self.label.setFont(self.label_font)
        self.label.setFixedSize(620, 100)

    def create_label_preds(self):
        self.preds_label = QLabel("~~Predictions~~")
        self.preds_label.setFixedSize(120, 40)

    def create_image_label(self):
        self.image = QPixmap(os.path.join('UI', 'fc_detect_image.png'))
        self.image_label = QLabel()
        self.image_label.setPixmap(self.image)
        self.image_label.setScaledContents(True)

    def create_input_line(self):
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Enter site url")
        self.input_line.setFixedSize(QSize(400, 40))

    def error_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setMinimumSize(QSize(300, 300))
        msg.setWindowTitle("Error message")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def create_combo_box(self):
        self.combo_box = QComboBox()
        self.combo_box.addItems(["CatBoost", "LightGBM"])   

    def create_layout(self):
        align_center = Qt.AlignmentFlag.AlignHCenter

        self.create_button()
        self.create_label_info()
        self.create_label_preds()
        self.create_image_label()
        self.create_input_line()
        self.create_combo_box()

        self.layout = QGridLayout()
        self.layout.setColumnStretch(5, 1)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.image_label, 0, 0, 1, 1, alignment=align_center)
        self.layout.addWidget(self.label, 1, 0, 1, 1, alignment=align_center)
        self.layout.addWidget(self.input_line, 2, 0, 1, 1, alignment=align_center)
        self.layout.addWidget(self.button, 3, 0, 1, 1, alignment=align_center)
        self.layout.addWidget(self.preds_label, 5, 0, 1, 1, alignment=align_center)
        self.layout.addWidget(self.combo_box, 3, 1, 1, 1, alignment=align_center)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

    def prediction_message(self, prediction, predict_proba, time):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(f"Prediction - {prediction}.\nProbability - {predict_proba}.\nExecution time - {round(time, 4)}")
        msg.setWindowTitle("Prediction")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def click_button(self):
        try:
            model_type = self.combo_box.currentText()
            url = self.input_line.text()
            start = time.time()
            prediction, predict_proba = self.app.predict(model_type, url)
            end = time.time()
            self.prediction_message(prediction, predict_proba, end - start)
            print('Prediction - ', prediction)
        except Exception as exc:
            self.error_message(str(exc))


class QTApp():
    def __init__(self) -> None:
        self.app = QApplication([])
        self.window = MainWindow(self)
        self.controller = Controller()

    def predict(self, model_type, url):
        return self.controller.predict(model_type, url)

    def run(self):
        self.window.show()
        self.app.exec()
