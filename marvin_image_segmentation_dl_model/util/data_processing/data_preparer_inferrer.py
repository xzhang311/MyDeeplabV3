#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch
import marvin_image_segmentation_dl_model.util.dataset as datasets
import marvin_image_segmentation_dl_model.util.data_processing.common as common


def get_data_loader(dataset,
                    dataset_root_folder,
                    batch_size=8,
                    workers = 32,
                    input_transform=None):
    test_set = datasets.__dict__[dataset](dataset_root_folder, is_train=False)
    data_loader = datasets.__dict__[dataset + '_loader']

    val_data_generator = common.DataGenerator(root=dataset_root_folder,
                                              path_list=test_set,
                                              input_transform=input_transform,
                                              loader=data_loader)

    # Create data loader, timeout set to be 30 secs.
    val_loader = torch.utils.data.DataLoader(
        val_data_generator, batch_size=batch_size,
        num_workers=workers, pin_memory=True, shuffle=False, timeout=30)

    return val_loader
