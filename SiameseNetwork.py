import tensorflow as tf

def ContrastiveLoss(y_true, y_pred):
    margin=1.

    positive_pair=tf.math.square(y_pred)
    negative_pair=tf.math.square(tf.math.maximum(margin-y_pred, 0.))
    loss=0.5 * ( y_true*positive_pair + ((1.-y_true)*negative_pair) )

    return tf.math.reduce_mean(loss)

class EuclideanDistance(tf.keras.layers.Layer):
    def call(self,inputs):
        return tf.expand_dims(tf.math.sqrt(tf.reduce_sum(tf.math.square(inputs[0]-inputs[1]),axis=1)),1)

class SiameseNetwork(tf.keras.Model):
    def __init__(
        self,
        feauture_extraction_layers=None,
        difference_layer=None,
        calculation_similar_layers=None
        ):
        super().__init__()

        if feauture_extraction_layers is None:
            self.feauture_extraction_layers=[
                tf.keras.applications.resnet50.ResNet50(include_top=False,weights="imagenet"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1024),
                tf.keras.layers.Dense(512,kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            ]
            self.feauture_extraction_layers[0].trainable=False
        else:
            self.feauture_extraction_layers=feauture_extraction_layers
        
        self.difference_layer=EuclideanDistance() if difference_layer is None else difference_layer

        self.calculation_similar_layers=[] if calculation_similar_layers is None else calculation_similar_layers

    def call(self,inputs):

        x1=inputs[:,0]
        x2=inputs[:,1]

        for layer in self.feauture_extraction_layers:
            x1=layer(x1)
            x2=layer(x2)
        
        x1=tf.keras.layers.Flatten()(x1)
        x2=tf.keras.layers.Flatten()(x2)

        x=self.difference_layer([x1,x2])

        for layer in self.calculation_similar_layers:
            x=layer(x)

        return x