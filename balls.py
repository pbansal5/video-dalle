import glob

import h5py
import torch
from torch.utils.data import Dataset

class BallsImage(Dataset):
    def __init__(self,root, mode):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'test']
        self.hf = h5py.File('%s/%s.hdf5'%(root,mode), 'r')
        self.hf = self.hf['X']

        self.sample_length = 100
        self.file_len = self.hf.shape[0]
        self.ep_len = self.hf.shape[1]

    def __getitem__(self, index):
        ep_num = index//self.ep_len
        index = index%self.ep_len
        video = self.hf[ep_num][index]
        video = torch.from_numpy(video).permute(2, 0, 1).float()
        return video

    def __len__(self):
        return self.file_len*self.ep_len

class Balls(Dataset):
    def __init__(self,root, mode):
        # path = os.path.join(root, mode)
        assert mode in ['train', 'val', 'test']
        self.hf = h5py.File('%s/%s.hdf5'%(root,mode), 'r')
        self.hf = self.hf['X']

        self.sample_length = 100
        self.file_len = self.hf.shape[0]
        self.ep_len = self.hf.shape[1]

    def __getitem__(self, index):
        video = self.hf[index][:self.sample_length]
        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        return video

    def __len__(self):
        return self.file_len

# class Balls(Dataset):
#     def __init__(self, root, mode):
#         # path = os.path.join(root, mode)
#         assert mode in ['train', 'val', 'test']
#         self.mode = mode
#         self.root = root
#         files = glob.glob(self.root + f'/*{mode}*')
#         assert len(files) > 0, f'No files for {mode} in {root}'

#         self.sample_length = 100
#         self.files = sorted(files)
#         # assume each file has same length
#         with h5py.File(files[0], 'r') as f:
#             self.file_len = len(f['imgs'])
#             self.ep_len = len(f['imgs'][0])

#     def __getitem__(self, index):
#         den = self.file_len
#         file_idx = index // den
#         index = index % den

#         with h5py.File(self.files[file_idx], 'r') as f:
#             video = f['imgs'][index][:self.sample_length]
#             video = torch.from_numpy(video).permute(0, 3, 1, 2)
#             video = video.float() / 255.0
#             return video

#     def __len__(self):
#         return self.file_len * len(self.files)


if __name__ == '__main__':
    b = Balls('/data/sdb/local/yifuwu2/datasets/video_transformer/4Balls_Mod_BouncingWall_1', 'train')
    b[0]
    print(len(b))
