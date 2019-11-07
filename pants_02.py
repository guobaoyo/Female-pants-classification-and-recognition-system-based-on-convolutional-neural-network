from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image
from keras.layers import *
from keras.models import *
from keras.applications.inception_resnet_v2 import preprocess_input
import inception_v4
from keras.optimizers import SGD
import numpy as np
import cv2

def pants_recognize(pic_path):
    width=480#图片的宽度，这个模型训练的时候就是取图片大小为480*480
    task_list={
        "pants_category":7
    }#设置pants_category为字典{}
    base_model = inception_v4.create_model(weights='imagenet', width=width, include_top=False)#调用inception_v4.py的create_model，并且不要最上面的全连接层，具体的在inceV4.py函数里面有解释
    input_tensor = Input((width, width, 3))#input为keras.layers的input函数
    x = input_tensor
    x = Lambda(preprocess_input, name='preprocessing')(x)
    #Lambda函数本身的意思是对上一层函数的输出施加任何theano/tensorflow表达式,lambda函数就是加了一个网络层给他起名字叫preprocessing
    # 然后调用keras.applications.inception_resnet_v2 的 preprocess_input对图片进行预处理，详情可以参考inceptionV4的def preprocess_input(x):
    #
    #且引用keras.applications.inception_resnet_v2的图像预处理对图片进行预处理。
    x = base_model(x)#传到base_model函数中，并且返回一个model
    x = GlobalAveragePooling2D()(x)#为刚才得出的model()空域信号施加一个全局平均值池化
    x = Dropout(0.5)(x)#d增加ropout有0.5的概率会“失活”
    x = [Dense(count, activation='softmax', name="recognize")(x) for name, count in task_list.items()]#怎么与前面的联系起来？
    #再添加全连接层，最后的这个全连接层接的激活函数是softmax函数
    model = Model(input_tensor, x)#
    model.load_weights('pants_inceptionv4_add1000_0.9258.h5')

    model.summary()
    #model.save('test1.h5')

    categories = ['紧身裤','锥形裤','直筒裤','喇叭裤','阔腿裤','灯笼裤','不可见']
    img_path= pic_path
    X_test=np.zeros((1,480,480,3),dtype=np.uint8)#设置X_test为1*480*480*3的4维张量，之所以第一个数字为1，是因为我们现在只有一张图片
    img=cv2.resize(cv2.imread(img_path),(480,480))#通过cv2这个函数将输入进来的图片进行调整大小到480*480
    X_test[0]=img[:,:,::-1]#第一个:取遍图像的所有行数。第二个:是取遍所有的列数。最后的::是取遍所有的通道数，-1是反向取值，则RGB变为BGR
    prediction = model.predict(X_test)#用model进行预测
    print(prediction)#输出7个数字，用softmax函数激活，和为1
    for i in range(7):
        if prediction[0][i]>0.7:
            print('此图片属于'+categories[i])
            return categories[i]

