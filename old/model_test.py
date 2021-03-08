import os
from PIL import Image
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
tf.random.set_seed(777) #하이퍼파라미터 튜닝을 위해 실행시 마다 변수가 같은 초기값 가지게 하기

##########데이터 로드

x_data = []
y_data = []

images = []
data_dir = '/home/mini/Documents/hand_re/hand_gesture_mini/hand_gestures/go_away'
file_names = os.listdir(data_dir)
file_names = [file_name for file_name in file_names if not file_name.startswith ('.')] #.DS_Store 제외
for file_name in file_names:
    file_path = '{}/{}'.format(data_dir, file_name)
    image = Image.open(file_path)
    image = image.resize((224, 224))
    numpy_image = np.array(image)
    images.append(numpy_image)
x_data.append(images)
y_data.append(0)

images = []
data_dir = '/home/mini/Documents/hand_re/hand_gesture_mini/hand_gestures/stop'
file_names = os.listdir(data_dir)
file_names = [file_name for file_name in file_names if not file_name.startswith ('.')] #.DS_Store 제외
for file_name in file_names:
    file_path = '{}/{}'.format(data_dir, file_name)
    image = Image.open(file_path)
    image = image.resize((224, 224))
    numpy_image = np.array(image)
    images.append(numpy_image)
x_data.append(images)
y_data.append(1)

images = []
data_dir = '/home/mini/Documents/hand_re/hand_gesture_mini/hand_gestures/basic'
file_names = os.listdir(data_dir)
file_names = [file_name for file_name in file_names if not file_name.startswith ('.')] #.DS_Store 제외
for file_name in file_names:
    file_path = '{}/{}'.format(data_dir, file_name)
    image = Image.open(file_path)
    image = image.resize((224, 224))
    numpy_image = np.array(image)
    images.append(numpy_image)
x_data.append(images)
y_data.append(2)

labels = [ 'go_away', 'stop_sign', 'basic' ]

##########데이터 분석

##########데이터 전처리

x_data = np.array(x_data)
x_train = x_data
x_test = x_data
y_train = y_data
y_test = y_data

# x_train = x_train / 255 #특성 스케일링
# x_test = x_test / 255

# labelr -> one-hot vector
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

if not os.path.exists('models/hand_gestures_classification_model'):
    os.makedirs('models/hand_gestures_classification_model')

##########모델 생성

base_model = hub.load('https://tfhub.dev/deepmind/i3d-kinetics-400/1')

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, base_model, trainable=False):
        super().__init__()

        self.base_model = base_model
        if trainable:
            self.base_model_trainable_variables = base_model.trainable_variables

    def call(self, input, training=False):

        return self.base_model.signatures['default'](input)['default']

input = tf.keras.layers.Input(shape=(None, 224, 224, 3))
net = MyLayer(base_model)(input)
net = tf.keras.layers.Dense(units=32, activation='relu')(net)
net = tf.keras.layers.Dense(units=len(labels), activation='softmax')(net)
model = tf.keras.models.Model(input, net)

print(model.summary())

##########모델 학습 및 검증


loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.Accuracy()
test_loss = tf.keras.metrics.Mean()
test_accuracy = tf.keras.metrics.Accuracy()

epochs = 50
best_test_loss = float('inf')
for epoch_index in range(epochs):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss_value = loss(y_train, predictions)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #print(len(model.trainable_variables)) #2

    train_loss.update_state(loss_value)
    train_accuracy.update_state(tf.argmax(y_train, axis=1), tf.argmax(predictions, axis=1))

    predictions = model(x_test)
    loss_value = loss(y_test, predictions)
    test_loss.update_state(loss_value)
    test_accuracy.update_state(tf.argmax(y_test, axis=1), tf.argmax(predictions, axis=1))

    print('epoch: {}/{}, train loss: {:.4f}, train accuracy: {:.4f}, test loss: {:.4f}, test accuracy: {:.4f}'.format(
        epoch_index + 1, epochs, train_loss.result().numpy(), train_accuracy.result().numpy(), test_loss.result().numpy(), test_accuracy.result().numpy()))

    if test_loss.result().numpy() < best_test_loss:
        tf.saved_model.save(model, 'models/hand_gestures_classification_model/saved_model') #최소 test_loss 마다 저장된 모델 저장
        best_test_loss = test_loss.result().numpy()

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

#'''
#내장 루프
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=[tf.keras.metrics.Accuracy()])

model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='models/hand_gestures_classification_model/saved_model', save_best_only=True, verbose=1)])
#'''
