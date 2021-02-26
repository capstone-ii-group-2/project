import cv2
import tensorflow as tf

# uncomment this to test your webcam
'''
cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)

if vc.isOpened(): # attempt to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow('preview', frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on escape key press
        break
cv2.destroyWindow("preview")
'''

# uncomment this to test tensorflow

mnist =  tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
