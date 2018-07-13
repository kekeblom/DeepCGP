import numpy as np
import tensorflow as tf
from unittest import TestCase
from .context import conv_gp
from conv_gp.views import RandomPartialView

class TestViews(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sess = tf.Session()

    def setUp(self):
        np.random.seed(0)

    def test_partial_view(self):
        view = RandomPartialView(input_size=(28, 28),
                filter_size=3, feature_maps=1, patch_count=5)

        self.assertEqual(len(view.patch_indices), 5)
        N = 10
        random_images = tf.constant(np.random.randn(N, 28, 28, 1))

        patches = view.extract_patches_PNL(random_images)
        patches = self.sess.run(patches)
        self.assertEqual(patches.shape, (5, 10, 9))

        slices = view.patch_indices[0]
        first_patch = self.sess.run(random_images[0, slices[0], slices[1], :])
        np.testing.assert_almost_equal(first_patch.ravel(), patches[0, 0, :].ravel())






