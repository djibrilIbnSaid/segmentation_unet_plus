from kerass.segmentation_models import Xnet
from tensorflow.keras.optimizers import Adam

model = Xnet(input_shape=(256, 256, 3), backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose')
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


model.save('model.h5')