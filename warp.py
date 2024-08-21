import numpy as np
from scipy.linalg import lstsq
from matplotlib import pyplot as plt
import tensorflow as tf

class Process():
	def __init__(self, imgSize, batchSize):
		self.batchSize = batchSize
		self.H, self.W = [int(x) for x in imgSize.split("x")]
		self.canon4pts = np.array([[-1, -1],
								   [-1, 1],
								   [1, 1],
								   [1, -1]], dtype=np.float32)
		self.image4pts = np.array([[0, 0],
								   [0, self.H - 1],
								   [self.W - 1, self.H - 1],
								   [self.W - 1, 0]], dtype=np.float32)
		self.refMtrx = self.fit(Xsrc=self.canon4pts, Xdst=self.image4pts)
		self.seed = 456
		# initial perturbation/translation scale
		self.pertScale = 0.25
		self.transScale = 0.25

	# fit (affine) warp between two sets of points
	def fit(self, Xsrc, Xdst):
		ptsN = len(Xsrc)
		X, Y = Xsrc[:, 0], Xsrc[:, 1]
		U, V = Xdst[:, 0], Xdst[:, 1]
		O, I = np.zeros([ptsN]), np.ones([ptsN])

		# least squares method: Ax = b
		A = np.concatenate((np.stack([X, Y, I, O, O, O], axis=1),
							np.stack([O, O, O, X, Y, I], axis=1)), axis=0)  # size of A = (2N, 6)
		b = np.concatenate((U, V), axis=0)  # size of b = (2N,)
		p1, p2, p3, p4, p5, p6 = lstsq(A, b)[0].squeeze()

		# expand the 2x3 affine transformation matrix into a 3x3 matrix in homogeneous coordinates.
		pMtrx = np.array([[p1, p2, p3],
						  [p4, p5, p6],
						  [0, 0, 1]], dtype=np.float32)
		return pMtrx

	# compute composition of warp parameters
	def compose(self, p, dp):
		pMtrx = self.vec2mtrx(p)
		dpMtrx = self.vec2mtrx(dp)
		pMtrxNew = tf.matmul(dpMtrx, pMtrx)  # size of pMtrxNew = (batchSize, 3, 3)
		pMtrxNew /= pMtrxNew[:, 2:3, 2:3]    # ensure that the last row of the matrix is indeed [0, 0, 1]
		pNew = self.mtrx2vec(pMtrxNew)
		return pNew

	# convert warp parameters to matrix
	def vec2mtrx(self, p):
		# size of p = (batchSize, 6)
		batchSize = p.get_shape()[0]
		O = tf.zeros([batchSize])
		I = tf.ones([batchSize])
		p1, p2, p3, p4, p5, p6 = tf.unstack(p, axis=1)

		# expand the 2x3 affine transformation matrix into a 3x3 matrix in homogeneous coordinates.
		pMtrx = tf.transpose(tf.stack([[I+p1, p2, p3],
									   [p4, I+p5, p6],
									   [O, O, I]]), perm=[2, 0, 1])
		return pMtrx

	# convert warp matrix to parameters
	def mtrx2vec(self, pMtrx):
		[row0, row1, _] = tf.unstack(pMtrx, axis=1)
		[e00, e01, e02] = tf.unstack(row0, axis=1)
		[e10, e11, e12] = tf.unstack(row1, axis=1)
		# affine
		p = tf.stack([e00 - 1, e01, e02, e10, e11 - 1, e12], axis=1)
		return p

	# warp the image
	def transformImage(self, image, pMtrx):
		batchSize = image.get_shape()[0]
		refMtrx = tf.tile(tf.expand_dims(self.refMtrx, axis=0), [batchSize, 1, 1])   # size of refMtrx = (batchSize, 3, 3)
		transMtrx = tf.matmul(refMtrx, pMtrx)

		# warp the canonical coordinates
		X, Y = np.meshgrid(np.linspace(-1, 1, self.W), np.linspace(-1, 1, self.H))        # size of X, Y = (W, H)
		X, Y = X.flatten(), Y.flatten()
		XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T                               # size of XYhom = (3, WxH)
		XYhom = np.tile(np.expand_dims(XYhom, axis=0), [batchSize, 1, 1]).astype(np.float32)    # size of XYhom = (batchSize, 3, WxH)
		XYwarpHom = tf.matmul(transMtrx, XYhom)                                                      # size of XYwarpHom = (batchSize, 3, WxH)
		XwarpHom, YwarpHom, ZwarpHom = tf.unstack(XYwarpHom, axis=1)                                 # size of XwarpHom = (batchSize, WxH)
		Xwarp = tf.reshape(XwarpHom / (ZwarpHom + 1e-8), [batchSize, self.H, self.W])           # ensure that the ZwarpHom = 1
		Ywarp = tf.reshape(YwarpHom / (ZwarpHom + 1e-8), [batchSize, self.H, self.W])

		# get the integer sampling coordinates
		Xfloor, Xceil = tf.floor(Xwarp), tf.math.ceil(Xwarp)
		Yfloor, Yceil = tf.floor(Ywarp), tf.math.ceil(Ywarp)
		XfloorInt, XceilInt = tf.cast(Xfloor, tf.int32), tf.cast(Xceil, tf.int32)
		YfloorInt, YceilInt = tf.cast(Yfloor, tf.int32), tf.cast(Yceil, tf.int32)
		imageIdx = np.tile(np.arange(batchSize).reshape([batchSize, 1, 1]), [1, self.H, self.W])
		imageVec = tf.reshape(image, [-1, int(image.shape[-1])])                              # size of imageVec = (batchSize * H * W, channels)
		imageVecOut = tf.concat([imageVec, tf.zeros([1, int(image.shape[-1])])], axis=0)      # size of imageVecOut = (batchSize * W * H + 1, channels) 處理超出邊界的情況
		idxUL = (imageIdx * self.H + YfloorInt) * self.W + XfloorInt
		idxUR = (imageIdx * self.H + YfloorInt) * self.W + XceilInt
		idxBL = (imageIdx * self.H + YceilInt) * self.W + XfloorInt
		idxBR = (imageIdx * self.H + YceilInt) * self.W + XceilInt
		idxOutside = tf.fill([batchSize, self.H, self.W], batchSize * self.H * self.W) # 處理超出邊界的情況

		def insideImage(Xint, Yint):
			return (Xint >= 0) & (Xint < self.W) & (Yint >= 0) & (Yint < self.H)

		# 將超出影像範圍的idx替換為 idxOutside，避免從影像中取樣時出現越界的情況。
		idxUL = tf.where(insideImage(XfloorInt, YfloorInt), idxUL, idxOutside)
		idxUR = tf.where(insideImage(XceilInt, YfloorInt), idxUR, idxOutside)
		idxBL = tf.where(insideImage(XfloorInt, YceilInt), idxBL, idxOutside)
		idxBR = tf.where(insideImage(XceilInt, YceilInt), idxBR, idxOutside)

		# bilinear interpolation
		# I(x,y)≈(1−dx)⋅(1−dy)⋅I(x1,y1)+dx⋅(1−dy)⋅I(x2,y1)+(1−dx)⋅dy⋅I(x1,y2)+dx⋅dy⋅I(x2,y2)
		Xratio = tf.reshape(Xwarp - Xfloor, [batchSize, self.H, self.W, 1])
		Yratio = tf.reshape(Ywarp - Yfloor, [batchSize, self.H, self.W, 1])
		imageUL = tf.cast(tf.gather(imageVecOut, idxUL), tf.float32) * (1 - Xratio) * (1 - Yratio)
		imageUR = tf.cast(tf.gather(imageVecOut, idxUR), tf.float32) * (Xratio) * (1 - Yratio)
		imageBL = tf.cast(tf.gather(imageVecOut, idxBL), tf.float32) * (1 - Xratio) * (Yratio)
		imageBR = tf.cast(tf.gather(imageVecOut, idxBR), tf.float32) * (Xratio) * (Yratio)
		imageWarp = imageUL + imageUR + imageBL + imageBR
		return imageWarp

	def genPerturbations(self, batchSize):
		# Initialize X, Y coordinate
		X = np.tile(self.canon4pts[:, 0], [batchSize, 1])     # size of X, Y = (batchSize, 4)
		Y = np.tile(self.canon4pts[:, 1], [batchSize, 1])

		# Generate perturbations
		dX = tf.random.normal([batchSize, 4], seed=self.seed) * self.pertScale + tf.random.normal([batchSize, 1], seed=self.seed) * self.transScale
		dY = tf.random.normal([batchSize, 4], seed=self.seed) * self.pertScale + tf.random.normal([batchSize, 1], seed=self.seed) * self.transScale

		O = np.zeros([batchSize, 4], dtype=np.float32)
		I = np.ones([batchSize, 4], dtype=np.float32)

		# fit warp parameters to generated displacements (affine)
		# J⋅p=dXY
		J = np.concatenate([np.stack([X, Y, I, O, O, O], axis=-1),
							np.stack([O, O, O, X, Y, I], axis=-1)], axis=1)
		dXY = tf.expand_dims(tf.concat([dX, dY], 1), -1)
		pPert = tf.compat.v1.matrix_solve_ls(J, dXY)[:, :, 0]
		return pPert

	def visualize_warp_process(self, images, imageWarpAll, pMtrxAll, warpN, save_path, pad_width=10, plot_firstN=5):

		def get_warpCoord(pMtrx):
			transMtrx = tf.matmul(self.refMtrx, pMtrx)
			# warp the canonical coordinates
			X, Y = np.meshgrid(np.linspace(-1, 1, self.W), np.linspace(-1, 1, self.H))  # size of X, Y = (W, H)
			X, Y = X.flatten(), Y.flatten()
			XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T  # size of XYhom = (3, WxH)
			XYwarpHom = tf.matmul(transMtrx, XYhom)  # size of XYwarpHom = (3, WxH)
			XwarpHom, YwarpHom, ZwarpHom = tf.unstack(XYwarpHom, axis=0)  # size of XwarpHom = (WxH, )
			Xwarp = tf.reshape(XwarpHom / (ZwarpHom + 1e-8), [self.H, self.W])  # ensure that the ZwarpHom = 1
			Ywarp = tf.reshape(YwarpHom / (ZwarpHom + 1e-8), [self.H, self.W])

			XwarpInt = tf.reshape(tf.cast(Xwarp, tf.int32), [784])
			YwarpInt = tf.reshape(tf.cast(Ywarp, tf.int32), [784])
			return XwarpInt, YwarpInt

		plot_firstN = np.minimum(images.shape[0], plot_firstN)
		for i in range(plot_firstN):
			fig, axes = plt.subplots(3, warpN, figsize=(10, 6))

			# Warp 前的影像
			axes[0, 0].imshow(images[i], cmap='gray')
			axes[0, 0].set_title(f'Original-{i}')

			ori_image = images[i].numpy().reshape((self.H, self.W)) * 255.
			ori_image = np.pad(ori_image, pad_width, 'constant', constant_values=255)
			for j in range(1, warpN + 1):

				axes[0, 1].imshow(imageWarpAll[0][i], cmap='gray')
				axes[0, 1].set_title(f'genPerturbations-{i}')
				for _ in range(warpN):
					axes[0, _].axis('off')

				Xwarp, Ywarp = get_warpCoord(pMtrxAll[j][i])
				x1, x2, x3, x4 = Xwarp[0], Xwarp[27], Xwarp[756], Xwarp[783]
				y1, y2, y3, y4 = Ywarp[0], Ywarp[27], Ywarp[756], Ywarp[783]

				img = tf.reshape(tf.cast(ori_image, tf.uint8), [self.H + pad_width * 2, self.W + pad_width * 2])

				axes[1, j-1].imshow(img, cmap="gray")
				axes[1, j-1].plot([x1 + pad_width, x2 + pad_width], [y1 + pad_width, y2 + pad_width], color='red')
				axes[1, j-1].plot([x2 + pad_width, x4 + pad_width], [y2 + pad_width, y4 + pad_width], color='red')
				axes[1, j-1].plot([x3 + pad_width, x4 + pad_width], [y3 + pad_width, y4 + pad_width], color='red')
				axes[1, j-1].plot([x3 + pad_width, x1 + pad_width], [y3 + pad_width, y1 + pad_width], color='red')

				# Warp 後的影像
				img_warp = imageWarpAll[j][i].numpy().reshape((self.H, self.W)) * 255.
				img_warp = tf.cast(tf.reshape(img_warp, [self.H, self.W]), tf.uint8)

				axes[2, j-1].imshow(img_warp, cmap='gray')
				axes[2, j-1].set_title(f'Warped Image {j}')
				axes[2, j-1].axis('off')


			plt.tight_layout()
			# plt.show()
			plt.savefig(save_path + f'-img_{str(i).zfill(3)}.png')
			plt.close()

