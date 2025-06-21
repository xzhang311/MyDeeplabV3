#!/apollo/sbin/envroot "$ENVROOT/bin/python"

import torch
import marvin_image_segmentation_dl_model.util.dataset as datasets
import marvin_image_segmentation_dl_model.util.data_processing.common as common

def get_data_loader(dataset,
                    dataset_root_folder,
                    manifest_file,
                    split=None,
                    batch_size = 2,
                    workers = 32,
                    input_transform=None,
                    target_transform = None,
                    co_transform_train=None,
                    co_transform_validation=None):
    train_set, test_set = datasets.__dict__[dataset](dataset_root_folder, manifest_file, is_train=True, split = split)
    data_loader = datasets.__dict__[dataset+'_loader']
    print(data_loader)
    train_data_generator = common.DataGenerator(root = dataset_root_folder,
                                          path_list = train_set,
                                          input_transform = input_transform,
                                          target_transform = target_transform,
                                          co_transform = co_transform_train,
                                          loader = data_loader)
    train_loader = torch.utils.data.DataLoader(
        train_data_generator, batch_size = batch_size,
        num_workers = workers, pin_memory = True, shuffle=True, timeout=30)


    val_data_generator = common.DataGenerator(root=dataset_root_folder,
                                              path_list=test_set,
                                              input_transform=input_transform,
                                              target_transform= target_transform,
                                              co_transform = co_transform_validation,
                                              loader=data_loader)

    # Create data loader, timeout set to be 30 secs.
    val_loader = torch.utils.data.DataLoader(
        val_data_generator, batch_size=batch_size,
        num_workers=workers, pin_memory=True, shuffle=False, timeout=30)

    return train_loader, val_loader
