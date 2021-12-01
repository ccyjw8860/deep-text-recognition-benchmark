from torch.utils.data import Dataset, DataLoader,  Subset
from torch._utils import _accumulate
import argparse

from dataset import OCRDataset, hierarchical_dataset_2, AlignCollate, hierarchical_dataset
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', help='path to training dataset',
                        default="D:/data/data_lmdb_release/training")
    parser.add_argument('--valid_data', help='path to validation dataset',
                        default="D:/data/data_lmdb_release/validation")
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, help='Transformation stage. None|TPS', default='TPS')
    parser.add_argument('--FeatureExtraction', type=str, help='FeatureExtraction stage. VGG|RCNN|ResNet', default='ResNet')
    parser.add_argument('--SequenceModeling', type=str, help='SequenceModeling stage. None|BiLSTM', default='BiLSTM')
    parser.add_argument('--Prediction', type=str, help='Prediction stage. CTC|Attn', default='Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')

    # """
    # CUSTOM DATASET
    # img size: torch.Size([192, 3, 32, 100])
    # """
    # _AlignCollate = AlignCollate(keep_ratio_with_pad=True)
    # _dataset, _dataset_log = hierarchical_dataset_2(root="D:/data/OCR_DATASET/train.zip")
    # _data_loader = DataLoader(_dataset, batch_size=192, shuffle=True, num_workers=0, collate_fn=_AlignCollate, pin_memory=True)
    # dataloader_list = []
    # dataloader_iter_list = []
    # dataloader_list.append(_data_loader)
    # dataloader_iter_list.append(iter(_data_loader))
    #
    # balanced_batch_images = []
    # balanced_batch_texts = []
    #
    # for i, dataloader_iter in enumerate(dataloader_iter_list):
    #     try:
    #         image, text = dataloader_iter.next()
    #         balanced_batch_images.append(image)
    #         balanced_batch_texts.append(text)
    #     except ValueError:
    #         pass
    # balanced_batch_images = torch.cat(balanced_batch_images, 0)
    # print(balanced_batch_images.size())

    dashed_line = '-' * 80
    print(dashed_line)
    print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
    assert len(opt.select_data) == len(opt.batch_ratio)

    _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    data_loader_list = []
    dataloader_iter_list = []
    batch_size_list = []
    Total_batch_size = 0
    for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
        _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
        print(dashed_line)
        _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
        total_number_dataset = len(_dataset)

        """
        The total number of data can be modified with opt.total_data_usage_ratio.
        ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
        See 4.2 section in our paper.
        """
        number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
        dataset_split = [number_dataset, total_number_dataset - number_dataset]
        indices = range(total_number_dataset)
        _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                       for offset, length in zip(_accumulate(dataset_split), dataset_split)]
        selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
        selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
        print(selected_d_log)
        batch_size_list.append(str(_batch_size))
        Total_batch_size += _batch_size

        _data_loader = torch.utils.data.DataLoader(
            _dataset, batch_size=_batch_size,
            shuffle=True,
            num_workers=int(opt.workers),
            collate_fn=_AlignCollate, pin_memory=True)
        data_loader_list.append(_data_loader)
        dataloader_iter_list.append(iter(_data_loader))

    Total_batch_size_log = f'{dashed_line}\n'
    batch_size_sum = '+'.join(batch_size_list)
    Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
    Total_batch_size_log += f'{dashed_line}'
    opt.batch_size = Total_batch_size

    print(Total_batch_size_log)

    balanced_batch_images = []
    balanced_batch_texts = []
    for i, data_loader_iter in enumerate(dataloader_iter_list):
        try:
            image, text = data_loader_iter.next()
            balanced_batch_images.append(image)
            balanced_batch_texts += text
        except StopIteration:
            dataloader_iter_list[i] = iter(data_loader_list[i])
            image, text = dataloader_iter_list[i].next()
            balanced_batch_images.append(image)
            balanced_batch_texts += text
        except ValueError:
            pass

    balanced_batch_images = torch.cat(balanced_batch_images, 0)

    print(balanced_batch_images.size())