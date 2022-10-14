import pandas as pd
import numpy as np
import tensorflow as tf
path = './iris.csv'

df = pd.read_csv(path)

df = df.reindex(np.random.permutation(len(df)))
print(df)

X, y = df.values[:, :-1], df.values[:, -1]

X = (X-X.mean())/X.std().astype('float32')


def label_encoder(y):
    if (y == "Setosa"):
        return 0.0
    if (y == "Versicolor"):
        return 1.0
    if (y == "Virginica"):
        return 2.0


y_classes = []
for l in y:
    y_classes.append(label_encoder(l))

# y = pd.DataFrame(y_classes)
y = y_classes

split_num = np.floor((len(X))*0.6).astype(int)

x_train, y_train = X[0:split_num], y[0:split_num]


x_test, y_test = X[split_num:-1], y[split_num:-1]

x_train = np.asarray(x_train).astype(float)

y_train = np.asarray(y_train).astype(int)

shape = x_train.shape[1]
print(shape)
print(x_train)
print(y_train)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(
        units=64, activation='relu', input_shape=(shape,)),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
metrics = [
    # tf.keras.metrics.sparse_categorical_accuracy(),
    tf.keras.metrics.Accuracy(),
    # tf.keras.metrics.AUC()
]

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy", metrics='accuracy')


model.fit(x_train, y_train, batch_size=50, epochs=50)
