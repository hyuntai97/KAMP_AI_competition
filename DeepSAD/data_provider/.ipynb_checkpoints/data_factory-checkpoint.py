from data_provider.data_loader import Dataset_Custom, Dataset_Pred, Dataset_Custom_TCN, Dataset_Custom_tr_split, Dataset_Pred2, Dataset_NAB, Dataset_Pred2_nab, Dataset_ASD
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
    'pred': Dataset_Pred,
    'pred2': Dataset_Pred2,
    'tcn': Dataset_Custom_TCN,
    'split': Dataset_Custom_tr_split,
    'nab': Dataset_NAB,
    'nab_pred2': Dataset_Pred2_nab,
    'asd': Dataset_ASD
}


def data_provider(data, timeenc, batch_size, root_path
                ,seq_len, data_path, num_workers, flag):
    Data = data_dict[data]
    timeenc = 0 if timeenc ==0 else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = batch_size

    data_set = Data(
        root_path=root_path,
        flag=flag,
        seq_len=seq_len,
        data_path=data_path,
        timeenc=timeenc,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)

    return data_set, data_loader