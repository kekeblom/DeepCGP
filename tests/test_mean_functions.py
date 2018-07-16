import numpy as np
import tensorflow as tf
from unittest import TestCase
from .context import conv_gp
from conv_gp.views import RandomPartialView, FullView
from conv_gp.mean_functions import PatchwiseConv2d, Conv2dMean

class TestMeanFunction(TestCase):
    def test_patchwise_conv(self):
        filter_size = 5
        patch_count = 9
        view = RandomPartialView((28, 28), filter_size, 1, patch_count)
        mean = PatchwiseConv2d(filter_size, 1, 3, 3)

        images = np.random.randn(10, 28, 28, 1)
        PNL_patches = view.extract_patches_PNL(images)
        sess = mean.enquire_session()
        mean_patches = sess.run(mean(PNL_patches))
        self.assertEqual(mean_patches.shape[0], 10)
        self.assertEqual(mean_patches.shape[1], 9)

    def test_full_patchwise_conv(self):
        filter_size = 5
        patch_count = 9
        view = FullView((28, 28), filter_size, 1)
        mean = PatchwiseConv2d(filter_size, 1, 24, 24)

        images = np.random.randn(10, 28, 28, 1)
        PNL_patches = view.extract_patches_PNL(images)
        sess = mean.enquire_session()


        mean_patches = sess.run(mean(PNL_patches))
        self.assertEqual(mean_patches.shape[0], 10)
        self.assertEqual(mean_patches.shape[1], 576)

        conv_mean = Conv2dMean(filter_size, 1)
        conv_mean_patches = sess.run(conv_mean(images))
        self.assertEqual(conv_mean_patches.shape, mean_patches.shape)


