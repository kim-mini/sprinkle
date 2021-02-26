images = []
data_dir = '/Users/mac/Downloads/hand_gestures/thumb_up'
file_names = os.listdir(data_dir)
file_names = [file_name for file_name in file_names if not file_name.startswith ('.')] #.DS_Store 제외
for file_name in file_names:
    file_path = '{}/{}'.format(data_dir, file_name)
    image = Image.open(file_path)
    image = image.resize((224, 224)) 
    numpy_image = np.array(image) 
    images.append(numpy_image)
x_test = np.array([images])
x_test = x_test / 255

y_predict = model.predict(x_test)

label = labels[y_predict[0].argmax()]
confidence = y_predict[0][y_predict[0].argmax()]
print(label, confidence)
