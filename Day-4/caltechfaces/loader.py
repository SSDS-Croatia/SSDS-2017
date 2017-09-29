import os
import numpy
import cv2

def get_sgm_tensor(nrows, ncols, r, c, s):
	#
	if r<0 or c<0:
		#
		return numpy.zeros((nrows, ncols), dtype=numpy.float32)
	#
	R = numpy.tile(numpy.linspace(0, nrows-1, nrows, dtype=numpy.float32), ncols).reshape(ncols, nrows).T
	C = numpy.tile(numpy.linspace(0, ncols-1, ncols, dtype=numpy.float32), nrows).reshape(nrows, ncols)
	#
	SGM = numpy.sqrt( (R-r)**2 + (C-c)**2 )
	#
	return numpy.exp(-SGM/s)

#
def get_loader(maxhw):
	#
	root = os.path.dirname(__file__)
	annots = open(os.path.join(root, 'annotations.txt'), 'r')
	imgpaths = []
	faces = []
	dict = {}
	for line in annots.readlines():
		#
		if line.strip() != '':
			imgname = line.split()[0]
			if imgname in dict:
				i = dict[imgname]
				faces[i].append([float(x) for x in line.split()[1:]])
			else:
				dict[imgname] = len(imgpaths)
				imgpaths.append(os.path.join(root, 'images', imgname))
				faces.append([[float(x) for x in line.split()[1:]]])
	#
	def load_sample(index=-1):
		#
		if index<0:
			index = numpy.random.randint(0, len(imgpaths))
			#print(index)
		#
		imgpath = imgpaths[index]
		coords = faces[index]
		#
		scalefactor = 1.0
		img = cv2.imread(imgpath)
		if img is None:
			return None, None
		if img.shape[0]>maxhw or img.shape[1]>maxhw:
			#print('* image too large? downscaling ...')
			scalefactor = numpy.min((maxhw/img.shape[0], maxhw/img.shape[1]))
		img = cv2.resize(img, (0,0), fx=scalefactor, fy=scalefactor)
		#
		tgt = numpy.zeros((3, img.shape[0], img.shape[1]), dtype=numpy.float32)
		for i in range(0, len(coords)):
			#
			ler = scalefactor*coords[i][1]
			lec = scalefactor*coords[i][0]
			rer = scalefactor*coords[i][3]
			rec = scalefactor*coords[i][2]
			nor = scalefactor*coords[i][5]
			noc = scalefactor*coords[i][4]
			#
			s = 0.25*( (ler-rer)**2 + (lec-rec)**2 )**0.5
			#
			tgt = tgt + numpy.stack([
				get_sgm_tensor(img.shape[0], img.shape[1], ler, lec, s),
				get_sgm_tensor(img.shape[0], img.shape[1], rer, rec, s),
				get_sgm_tensor(img.shape[0], img.shape[1], nor, noc, s)
			])
		#
		img = img.astype(numpy.float32)/255.0
		tgt = numpy.transpose(tgt, (1, 2, 0))
		#
		IMG = numpy.zeros((int(maxhw), int(maxhw), 3), dtype=numpy.float32)
		TGT = numpy.zeros((int(maxhw), int(maxhw), 3), dtype=numpy.float32)
		#
		IMG[0:img.shape[0], 0:img.shape[1], :] = img
		TGT[0:img.shape[0], 0:img.shape[1], :] = tgt
		#
		return IMG, TGT
	#
	def load_batch(n=32):
		imgs = []
		tgts = []
		for i in range(0, n):
			img, tgt = load_sample()
			imgs.append(img)
			tgts.append(tgt)
		return numpy.stack(imgs), numpy.stack(tgts)
	#
	return load_batch

#
'''
load_sample = get_loader()
img, tgt = load_sample()
cv2.imwrite('img.jpg', 255*img)
cv2.imwrite('tgt.jpg', 255*(tgt[:, :, 0] + tgt[:, :, 1] + tgt[:, :, 2]))
#'''