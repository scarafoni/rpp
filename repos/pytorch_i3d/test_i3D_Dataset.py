from unittest import TestCase
from i3d_feats_dataset import I3D_Dataset


class TestI3D_Dataset(TestCase):
    def test__window_from_fname(self):
        fname = '/coc/pcba1/dscarafoni3/NRI/NRI_Data/cropped_rgb_data/i3d_features/GOPR0059/i3d_features_223-255.npy'
        starti, endi = I3D_Dataset._window_from_fname(fname)
        self.assertEqual(starti, 223)
        self.assertEqual(endi, 255)
