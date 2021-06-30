from torch.utils.data.dataloader import default_collate


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __call__(self, batch):
        batch_data = dict()
        for key in batch[0].keys():
            if key == 'img_meta':
                batch_data[key] = [b[key] for b in batch]
                continue
            batch_data[key] = default_collate([b[key] for b in batch])
        return batch_data