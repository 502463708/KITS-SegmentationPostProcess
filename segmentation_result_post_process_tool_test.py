import argparse
import os
import SimpleITK as sitk

from segmentation_result_post_process_tool import PredictionMaskPostProcessTool


def ParseArguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_mask_path',
                        type=str,
                        default='/data/lars/projects/KITS-SegmentationPostProcess/prediction_00266.nii',
                        help='The absolute path of the prediction result (mask format).')
    parser.add_argument('--kidney_volume_threshold',
                        type=int,
                        default=50000,
                        help='The volume threshold determining whether a kidney candidate can be discarded.')
    parser.add_argument('--keep_max_two_kidney',
                        type=bool,
                        default=True,
                        help='The flag indicating whether keep the max two kidney connected components.')

    args = parser.parse_args()

    assert os.path.exists(args.prediction_mask_path), '{} does not exist.'.format(args.prediction_result_path)
    assert args.kidney_volume_threshold >= 0

    return args


def TestPredictionMaskPostProcessTool(args):
    prediction_mask_image = sitk.ReadImage(args.prediction_mask_path, sitk.sitkUInt8)

    prediction_mask_array = sitk.GetArrayFromImage(prediction_mask_image)

    post_process_tool_obj = PredictionMaskPostProcessTool(args.kidney_volume_threshold)
    post_processed_mask_array = post_process_tool_obj.run(prediction_mask_array, args.keep_max_two_kidney)

    post_processed_mask_image = sitk.GetImageFromArray(post_processed_mask_array)
    post_processed_mask_image.SetDirection(prediction_mask_image.GetDirection())
    post_processed_mask_image.SetSpacing(prediction_mask_image.GetSpacing())
    post_processed_mask_image.SetOrigin(prediction_mask_image.GetOrigin())

    sitk.WriteImage(post_processed_mask_image,
                    args.prediction_mask_path.replace('prediction', 'post_processed_prediction'))

    return


if __name__ == '__main__':
    args = ParseArguments()

    TestPredictionMaskPostProcessTool(args)
