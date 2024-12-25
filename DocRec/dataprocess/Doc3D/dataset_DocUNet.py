from torch.utils.data import Dataset


class DocUNetDataset(Dataset):

    def __init__(self, seg_path="/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/seg/final",
                 grid_path="", res_path="/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/scan"):
        super(DocUNetDataset, self).__init__()

        data = []

    def __len__(self):
        return