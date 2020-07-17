import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from scipy.misc import imsave
from scipy.misc import imresize
from sklearn.manifold import TSNE

class Plot_Reproduce_Performance():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, resize_factor=1.0):
        self.DIR = DIR

        assert n_img_x > 0 and n_img_y > 0

        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_tot_imgs = n_img_x * n_img_y

        assert img_w > 0 and img_h > 0

        self.img_w = img_w
        self.img_h = img_h

        assert resize_factor > 0

        self.resize_factor = resize_factor

    def save_images(self, images, name='result.jpg'):
        images = images.reshape(self.n_img_x * self.n_img_y, 3, self.img_h , self.img_w)
        images = images.transpose(0, 2, 3, 1)
        store_image = self._merge(images, [self.n_img_y, self.n_img_x])
        imsave(self.DIR + "/" + name, store_image)

    #进行图片拼接
    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]         #size 为 32 * 32

        h_ = int(h * self.resize_factor)
        w_ = int(w * self.resize_factor)

        img = np.zeros((h_ * size[0], w_ * size[1], 3))     #size 为320 * 320 * 3

        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])

            image_ = imresize(image, size=(w_, h_, 3), interp='bicubic')        #image_size is：32 * 32 *3

            img[j * h_:j * h_ + h_, i * w_:i * w_ + w_, :3] = image_

        return img



def plot_t_sne(z, images):
    n_samples = images.shape[0]
    images = images.transpose(0, 2, 3, 1)
    z_embedded = TSNE(n_components=2).fit_transform(z)
    print('input_embedded:', z_embedded.shape)

    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(n_samples):
        dist = np.sum((z_embedded[i] - shown_images) ** 2, 1)
        if np.min(dist) < 8e-3:
            #do not show points that are too close
            continue
        shown_images = np.r_[shown_images, [z_embedded[i]]]
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i]), z_embedded[i])
        ax.add_artist(imagebox)

    plt.savefig(fname='latent_space.jpg', format='jpg')
    plt.show()


