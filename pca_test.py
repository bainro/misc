import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Perform PCA on the activations
def plot_scree(activations, title):
    pca = PCA(n_components=35)
    pca.fit(activations)
    explained_variance = pca.explained_variance_ratio_

    plt.plot(np.cumsum(explained_variance), label=title)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
class validation_threshold(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(validation_threshold, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_acc = logs["val_accuracy"]
        if val_acc >= self.threshold:
            self.model.stop_training = True
    
def create_model():
    # Feature Extraction is performed by ResNet50 pretrained on imagenet weights. 
    def feature_extractor(inputs):
    
      feature_extractor = tf.keras.applications.resnet.ResNet50(input_shape=(224, 224, 3),
                                                                            include_top=False,
                                                                            weights='imagenet')(inputs)
      return feature_extractor
    
    # Defines final dense layers and subsequent softmax layer for classification.
    def classifier(inputs):
        x = tf.keras.layers.Identity()(inputs) # for intermediate activations
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dense(100, activation="softmax", name="classification")(x)
        return x
    
    '''
    Since input image size is (32 x 32), first upsample the image by factor of (7x7) to transform it to (224 x 224)
    Connect the feature extraction and "classifier" layers to build the model.
    '''
    def final_model(inputs):
    
        resize = tf.keras.layers.UpSampling2D(size=(7,7))(inputs)
    
        resnet_feature_extractor = feature_extractor(resize)
        classification_output = classifier(resnet_feature_extractor)
    
        return classification_output
    
    '''
    Define the model and compile it. 
    Use Stochastic Gradient Descent as the optimizer.
    Use Sparse Categorical CrossEntropy as the loss function.
    '''
    def define_compile_model():
      inputs = tf.keras.layers.Input(shape=(32,32,3))
      
      classification_output = final_model(inputs) 
      model = tf.keras.Model(inputs=inputs, outputs = classification_output)
     
      model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics = ['accuracy'])
      
      return model
    
    model = define_compile_model()
    #model.summary()
    return model

(_x_train, y_train), (_x_test, y_test) = cifar100.load_data()

def preprocess_image_input(input_images):
  input_images = input_images.astype('float32')
  output_ims = tf.keras.applications.resnet50.preprocess_input(input_images)
  return output_ims
x_train = preprocess_image_input(_x_train)
x_test = preprocess_image_input(_x_test)

# Function to filter dataset by classes
def filter_by_classes(x, y, classes):
    mask = np.isin(y, classes)
    mask = mask.squeeze()
    return x[mask], y[mask]

# Filter datasets
x_train_15classes, y_train_15classes = filter_by_classes(x_train, y_train, range(15))
x_test_15classes, y_test_15classes = filter_by_classes(x_test, y_test, range(15))

x_train_20classes, y_train_20classes = filter_by_classes(x_train, y_train, range(20))
x_test_20classes, y_test_20classes = filter_by_classes(x_test, y_test, range(20))

x_train_25classes, y_train_25classes = filter_by_classes(x_train, y_train, range(25))
x_test_25classes, y_test_25classes = filter_by_classes(x_test, y_test, range(25))

# Train models
bs = 64
max_eps = 150
val_callback = validation_threshold(threshold=0.8)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=6,
                                            restore_best_weights=True)
datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip = True)
datagen.fit(x_train_15classes)
model_15classes = create_model()
model_15classes.fit(datagen.flow(x_train_15classes, y_train_15classes, batch_size=bs), 
                              epochs=max_eps, 
                              validation_data=(x_test_15classes, y_test_15classes), 
                              verbose=1, 
                              callbacks=[callback, val_callback])

plt.figure()
intermediate_layer_model = tf.keras.Model(inputs=model_15classes.input, 
                                          outputs=model_15classes.get_layer('identity').output)
activations_15classes = intermediate_layer_model.predict(x_test_25classes[:1000, ...])

plot_scree(activations_15classes.reshape(1000, 7*7*2048), 'N=15')
del activations_15classes # clear up RAM
tf.keras.backend.clear_session() # clear up GPU VRAM

del callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=6, 
                                            restore_best_weights=True)

datagen.fit(x_train_20classes)
model_20classes = create_model()
model_20classes.fit(datagen.flow(x_train_20classes, y_train_20classes, batch_size=bs), 
                              epochs=max_eps, 
                              validation_data=(x_test_20classes, y_test_20classes), 
                              verbose=1, 
                              callbacks=[callback, val_callback])

intermediate_layer_model = tf.keras.Model(inputs=model_20classes.input, 
                                          outputs=model_20classes.get_layer('identity').output)
activations_20classes = intermediate_layer_model.predict(x_test_25classes[:1000, ...])
plot_scree(activations_20classes.reshape(1000, 7*7*2048), 'N=20')
del activations_20classes # clear up RAM
tf.keras.backend.clear_session() # clear up GPU VRAM

del callback
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=6, 
                                            restore_best_weights=True)

datagen.fit(x_train_25classes)
model_25classes = create_model()
model_25classes.fit(datagen.flow(x_train_25classes, y_train_25classes, batch_size=bs), 
                              epochs=max_eps, 
                              validation_data=(x_test_25classes, y_test_25classes), 
                              verbose=1, 
                              callbacks=[callback, val_callback])

intermediate_layer_model = tf.keras.Model(inputs=model_25classes.input, 
                                          outputs=model_25classes.get_layer('identity').output)
activations_25classes = intermediate_layer_model.predict(x_test_25classes[:1000, ...])
plot_scree(activations_25classes.reshape(1000, 7*7*2048), 'N=25')
plt.legend()
plt.show()