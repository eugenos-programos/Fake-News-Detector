from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QLineEdit, QGridLayout, QWidget, QComboBox
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QSize, Qt

import typing


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

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
        self.image = QPixmap('fc_detect_image.png')
        self.image_label = QLabel()
        self.image_label.setPixmap(self.image)
        self.image_label.setScaledContents(True)

    def create_input_line(self):
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Enter site url")
        self.input_line.setFixedSize(QSize(400, 40))

    def create_combo_box(self):
        self.combo_box = QComboBox()
        self.combo_box.addItems(["BERT", "CatBoost", "LightGBM"])   

    def create_layout(self):
        align_center = Qt.AlignmentFlag.AlignHCenter

        self.create_button()
        self.create_label_info()
        self.create_label_preds()
        self.create_image_label()
        self.create_input_line()
        self.create_combo_box()

        layout = QGridLayout()
        layout.setColumnStretch(5, 1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_label, 0, 0, 1, 1, alignment=align_center)
        layout.addWidget(self.label, 1, 0, 1, 1, alignment=align_center)
        layout.addWidget(self.input_line, 2, 0, 1, 1, alignment=align_center)
        layout.addWidget(self.button, 3, 0, 1, 1, alignment=align_center)
        layout.addWidget(self.preds_label, 5, 0, 1, 1, alignment=align_center)
        layout.addWidget(self.combo_box, 3, 1, 1, 1, alignment=align_center)

        self.container = QWidget()
        self.container.setLayout(layout)
        
        self.setCentralWidget(self.container)

    def click_button(self):
        print(self.combo_box.currentText()) 


def run():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()