from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D
from keras.layers import UpSampling2D, Dropout, BatchNormalization, Conv2DTranspose

def conv_block(m, dim, activ_func, batch_norm, res, dropout=0):
	n = Conv2D(dim, 3, activation=activ_func, padding='same') (m)
	n = BatchNormalization()(n) if batch_norm else n
	n = Dropout(dropout)(n) if dropout else n
	n = Conv2D(dim, 3, activation=activ_func, padding='same')(n)
	n = BatchNormalization()(n) if batch_norm else n
	return Concatenate()([m,n]) if res else n

def level_block(m, dim, depth, inc, activ_func, dropout, batch_norm, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, activ_func, batch_norm, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, activ_func, dropout, batch_norm, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=activ_func, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=activ_func, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, activ_func, batch_norm, res)
	else:
		m = conv_block(m, dim, activ_func, batch_norm, res, dropout)
	return m

def UNet(img_shape, out_ch=4, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.5, batch_norm=True, maxpool=True, upconv=True, residual=False):
	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batch_norm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='softmax')(o)
	return Model(inputs=i, outputs=o)