import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from pants_02 import pants_recognize
a = []
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.resize(480, 503)#主窗口的大小
        self.setWindowTitle("基于InceptionV4卷积神经网络模型的女裤识别系统")#界面的名字
        self.label_school = QLabel(self)
        self.label_result = QLabel(self)
        self.label_school.setText("")#待显想·示的图片名字
        #self.label.setText("山东科技大学计算机科学与工程学院网络工程15-1石成功毕业设计展示界面")#待显示的图片名字
        self.label_school.setFixedSize(480, 480)
        self.label_school.move(1, 1)
        self.label_school.setStyleSheet("QLabel{background:white;}" "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}" )
        self.label_result.setStyleSheet("QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}" )
        self.label_college = QLabel(self)

        self.label_school1 = QLabel(self)
        self.label_school1.setText("                            ")
        self.label_school1.move(1, 480)
        self.label_school1.setStyleSheet("QLabel{background-color:rgb(300, 300, 300，120);}" "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}")


        self.label_college.setText("计算机科学与工程学院           ")
        self.label_college.move(1,510)
        self.label_college.setStyleSheet("QLabel{background-color:rgb(300,300,300);}" "QLabel{color:rgb(300,300,300,120);font-size:30px;font-weight:bold;font-family:宋体;}" )
        self.label_specialty = QLabel(self)
        self.label_specialty.setText("网络工程2015-1                              ")
        self.label_specialty.move(1,540)
        self.label_specialty.setStyleSheet("QLabel{background-color:rgb(224, 255, 255);}" "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}" )

        self.label_name = QLabel(self)
        self.label_name.setText("石成功                                      ")
        self.label_name.move(1, 560)
        self.label_name.setStyleSheet("QLabel{background-color:rgb(224, 255, 255);}" "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}")

        self.label_specialty = QLabel(self)
        self.label_specialty.setText("基于InceptionV4卷积神经网络模型的女裤识别系统")
        self.label_specialty.move(1, 580)
        self.label_specialty.setStyleSheet("QLabel{background-color:rgb(224, 255, 255);}" "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}")


        self.label_specialty = QLabel(self)
        self.label_specialty.setText("----->")
        self.label_specialty.move(108, 483)
        self.label_specialty.setStyleSheet("QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")


        self.label_specialty = QLabel(self)
        self.label_specialty.setText("----->")
        self.label_specialty.move(225, 483)
        self.label_specialty.setStyleSheet("QLabel{color:rgb(300,300,300,120);font-size:15px;font-weight:bold;font-family:宋体;}")



        btn = QPushButton(self)
        btn.setText("choose the picture")
        btn.move(1, 480)
        btn.clicked.connect(self.openimage)
        #run,底下的4行还没写好
        btn1 = QPushButton(self)
        btn1.setText("run")
        btn1.move(163, 480)
        btn1.clicked.connect(self.run)
#此程序中曾经出现的难题：
#1、没定义全局变量，导致openimage函数得出来的文件路径无法被run()函数接收到
#2、btn1.clicked.connect()括号里面必须接上self.run  而不能是self.run(),因为如果这样的话在my = picture()会跳过self.openimage直接到self.run()这样是不合理的、
    def openimage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print(imgName)#imgName = D:/tensorboard-shishi/test/data/picture_neversee/testpic.jpg
        global imgName1
        imgName1 = imgName#jieguo = pants_recognize(imgName)
        print(imgName1)
        # print('pants_02返回的结果为'+ jieguo)
        jpg = QtGui.QPixmap(imgName).scaled(self.label_school.width(), self.label_school.height())
        self.label_school.setPixmap(jpg)#将图片显示出来
    def run(self):
        #imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        global jieguo
        jieguo = pants_recognize(imgName1)
        print(imgName1)
        print(jieguo)
        print('pants_02返回的结果为' + jieguo)
        jieguo = str('   此类裤型为：')+jieguo+str('     .                 ')
        self.label_result.setText(jieguo)
        self.label_result.move(255,480)
        self.label_result.setFixedSize(210, 20)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())

