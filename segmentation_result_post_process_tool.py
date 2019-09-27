import numpy as np

from skimage import measure


class PredictionMaskPostProcessTool(object):
    """
    This class encapsulates a variety of functions which implement post-process for
    predicted kidney segmentation result (mask format). And the data type of the
    mask required here is np.ndarray
    """

    def __init__(self, kidney_volume_threshold=50000):
        """
        :param kidney_volume_threshold: kidney candidate whose volume < kidney_volume_threshold
        will be discarded.
        """

        assert kidney_volume_threshold >= 0

        self.kidney_volume_threshold = kidney_volume_threshold

        return

    def remove_outlier_kidney(self, input_mask, keep_max_two_kidney=False):
        """
        This function can remove tiny kidney false positives.
        :param input_mask: 0->background, 1->kidney, 2->tumor.
        :param keep_max_two_kidney:
        :return: output_mask: 0->background, 1->kidney, 2->tumor.
        """
        assert isinstance(input_mask, np.ndarray)
        assert len(input_mask.shape) == 3
        assert input_mask.min() >= 0
        assert input_mask.max() <= 2

        # generate a binary mask for kidney only
        kidney_mask = np.zeros_like(input_mask)
        kidney_mask[input_mask == 1] = 1

        # extract kidney connected components and generate its properties
        kidney_connected_components = measure.label(kidney_mask, connectivity=2)
        kidney_connected_components_props = measure.regionprops(kidney_connected_components)

        # save the retained connected components
        area_2_connect_idx_dict = dict()

        # remove tiny kidney candidate
        for connect_idx in range(1, kidney_connected_components.max() + 1):
            if kidney_connected_components_props[connect_idx - 1].area < self.kidney_volume_threshold:
                kidney_mask[kidney_connected_components == connect_idx] = 0
            else:
                area_2_connect_idx_dict[connect_idx] = kidney_connected_components_props[connect_idx - 1].area

        # keep the max 2 kidney
        while keep_max_two_kidney and len(area_2_connect_idx_dict.keys()) > 2:
            # fine the connected component with the min area to be removed
            min_connected_component_area = np.array(list(area_2_connect_idx_dict.values())).min()

            # find its corresponding connect_idx
            connect_idx = 0
            while area_2_connect_idx_dict[connect_idx] != min_connected_component_area:
                connect_idx += 1

            # remove this connected component from kidney_mask
            kidney_mask[kidney_connected_components == connect_idx] = 0
            del area_2_connect_idx_dict[connect_idx]

        # generate output_mask
        output_mask = np.zeros_like(input_mask)
        output_mask[kidney_mask == 1] = 1
        output_mask[input_mask == 2] = 2

        return output_mask

    def remove_outlier_tumor(self, input_mask):
        """
        This function can remove the tumors disconnected with kidney.
        :param input_mask: 0->background, 1->kidney, 2->tumor.
        :return: output_mask: 0->background, 1->kidney, 2->tumor.
        """
        assert isinstance(input_mask, np.ndarray)
        assert len(input_mask.shape) == 3
        assert input_mask.min() >= 0
        assert input_mask.max() <= 2

        # regard both kidney and tumor as foreground
        foreground_mask = np.zeros_like(input_mask)
        foreground_mask[input_mask != 0] = 1

        # generate a binary mask for tumor only
        tumor_mask = np.zeros_like(input_mask)
        tumor_mask[input_mask == 2] = 1

        # extract foreground connected components
        foreground_connected_components = measure.label(foreground_mask, connectivity=2)

        # remove disconnected tumors
        for connect_idx in range(1, foreground_connected_components.max() + 1):
            if ((foreground_connected_components == connect_idx) * (input_mask == 1)).sum() == 0:
                tumor_mask[foreground_connected_components == connect_idx] = 0

        # generate output_mask
        output_mask = np.zeros_like(input_mask)
        output_mask[input_mask == 1] = 1
        output_mask[tumor_mask == 1] = 2

        return output_mask

    def run(self, input_mask, keep_max_two_kidney=False):

        return self.remove_outlier_tumor(self.remove_outlier_kidney(input_mask, keep_max_two_kidney))
