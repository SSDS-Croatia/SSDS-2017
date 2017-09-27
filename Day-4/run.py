import tensorflow as tf
import numpy
import cv2
import sys

#
# load network
#

maxhw = 160

import tensorflow as tf
from models import *
x = tf.placeholder(tf.float32, [None, maxhw, maxhw, 3])
pred, is_training = build_model(x, 3) # `x` is input, `3` specifies the number of output channels

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, 'save/model')

#
# runtime
#

def process_image(img, thr=0.1):
	#
	X = numpy.stack([img])
	lbl = sess.run(pred, feed_dict={x: X, is_training: False})
	lbl = lbl[0, :, :, :]
	# threshold array
	lbl[lbl<thr] = 0
	return lbl

#
if len(sys.argv)>=2:
	#
	img = cv2.imread(sys.argv[1])
	scalefactor = 1.0
	if img.shape[0]>maxhw or img.shape[1]>maxhw:
		scalefactor = numpy.min((maxhw/img.shape[0], maxhw/img.shape[1]))
	img = cv2.resize(img, (0,0), fx=scalefactor, fy=scalefactor)
	#
	IMG = numpy.zeros((int(maxhw), int(maxhw), 3), dtype=numpy.float32)
	IMG[0:img.shape[0], 0:img.shape[1], :] = img.astype(numpy.float32)/255.0
	#
	LBL = process_image(IMG, thr=0.1)

	cv2.imwrite('img.jpg', 255*IMG)
	cv2.imwrite('lbl.jpg', 255*LBL)
else:
	cap = cv2.VideoCapture(0)
	while(True):
		#
		ret, frm = cap.read()
		#
		if frm.shape[0]>maxhw or frm.shape[1]>maxhw:
			scalefactor = numpy.min((maxhw/frm.shape[0], maxhw/frm.shape[1]))
		frm = cv2.resize(frm, (0,0), fx=scalefactor, fy=scalefactor)
		#
		FRM = numpy.zeros((int(maxhw), int(maxhw), 3), dtype=numpy.float32)
		FRM[0:frm.shape[0], 0:frm.shape[1], :] = frm.astype(numpy.float32)/255.0
		#
		lbl = process_image(frm, thr=0.1)
		#
		cv2.imshow('frm', 255*FRM)
		cv2.imshow('sgm', 255*LBL)
		#
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break
	#
	cap.release()
	cv2.destroyAllWindows()
