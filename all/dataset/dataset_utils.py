import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import torch
from tqdm import tqdm
import h5py
from pathlib import Path
import cv2

from utils.utils import set_seed
from dataset.RandomResizedCropCustom import RandomResizedCropCustom


def get_dataset(args, train):
    if args.dataset == 'rlv':
        return RLVDataset(window=args.num_frames, flow=False)
    elif args.dataset == 'youtube_small':
        return YouTubeSmallDataset(window=args.num_frames, flow=False)
    elif args.dataset == 'fight_surv':
        return FightSurvDataset(window=args.num_frames, flow=False, train=train)
    elif args.dataset == 'rwf':
        return RWFDataset(window=args.num_frames, train=train, stride=0)
    elif args.dataset == 'rwf_npy':
        if train:
            return RWFDataset_npy(data_dir='/home/qzt/data/RWF_2000_npy/train', data_partition='train', clip_len=30, temporal_stride=-1)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def create_dataloader(batch_size, dataset, ratio=0.85, num_worker=8):
    set_seed(42)

    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(len(dataset)*ratio), len(dataset)-int(len(dataset)*ratio)])

    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    val_loader = DataLoader(
        dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return train_loader, val_loader


# window: how many frames in a desired clip
# stride: how many frames as interval to select the next frame to assemble into the desired clip


class RLVDataset(Dataset):
    def __init__(self, window, flow):
        self.root = r'/home/shared/action/Data/RLV/'
        self.flow = flow
        self.window = window
        self.data = []
        self.preload = {}
        preload_frac = 0
        for clip in glob.glob(self.root+"processed/*"):

            if 'flow' in clip:
                continue
            if "_h5" not in clip:
                continue
            base = clip.split("/")[:-1]
            fname = clip.split("/")[-1]
            fname_root = fname.split('.')[0]
            label = "NV_" not in fname_root  # True if violence

            # Dataset should be about 500GB, 1 node has 256 gb ram. can preload if we find a way to get more memory
            img = h5py.File(clip, 'r')['video']
            #flow = np.load(base+f"/{fname_root}_flow.npy")

            frame_num, w, h, c = img.shape

            self.data.append((frame_num, clip, label))

        self.clips = []

        print("Calculating Clips")
        for video in self.data:
            frame_num, path, label = video

            for frame_offset in range(frame_num-window+1):
                self.clips.append((path, frame_offset, label))

        print("Preloading RLV")
        for item in tqdm(sorted(self.data, key=lambda x: x[0], reverse=True)[:int(len(self.data)*preload_frac)]):
            frame_num, video_path, label = item
            self.preload[video_path] = h5py.File(video_path, 'r')['video']

    def __getitem__(self, idx):
        path, frame_offset, label = self.clips[idx]

        img = self.preload.get(path)
        if img is None:
            img = h5py.File(path, 'r')['video']

        base = self.root + "processed"
        fname_root = path.split("/")[-1].split('.')[0]

        img_data = torch.Tensor(
            img[frame_offset:frame_offset+self.window]).permute(0, 3, 1, 2)  # window, c(3), h, w
        if self.flow:
            flow = np.load(base+f"/{fname_root}_flow.npy")
            flow_data = torch.Tensor(
                flow[frame_offset:frame_offset+self.window-1]).permute(0, 3, 1, 2)  # window-1, c(2), h, w
            return img_data, flow_data, torch.Tensor([label])
        else:
            return img_data, torch.Tensor([label])

    def __len__(self):
        return len(self.clips)


class YouTubeSmallDataset(Dataset):
    def __init__(self, window, flow):
        self.root = r'/home/shared/action/Data/youtube_small/'

        self.window = window
        self.data = []
        self.flow = flow
        for clip in glob.glob(self.root + "processed/*"):

            if 'flow' in clip:
                continue
            if "_h5" not in clip:
                continue
            base = clip.split("/")[:-1]
            fname = clip.split("/")[-1]
            fname_root = fname.split('.')[0]
            # print(f"Clip: {fname_root}")
            label = 'fi' == fname_root[:2]  # True if violence

            # Dataset should be about 500GB, 1 node has 256 gb ram. can preload if we find a way to get more memory
            img = h5py.File(clip, 'r')['video']
            # flow = np.load(base+f"/{fname_root}_flow.npy")

            frame_num, w, h, c = img.shape

            self.data.append((frame_num, clip, label))

        self.clips = []

        print("Calculating Clips")
        for video in self.data:
            frame_num, path, label = video

            for frame_offset in range(frame_num - window + 1):
                self.clips.append((path, frame_offset, label))

    def __getitem__(self, idx):
        path, frame_offset, label = self.clips[idx]

        img = h5py.File(path, 'r')['video']

        base = self.root + "processed"
        fname_root = path.split("/")[-1].split('.')[0]

        img_data = torch.Tensor(
            img[frame_offset:frame_offset + self.window]).permute(0, 3, 1, 2)  # window, c(3), h, w
        if self.flow:
            flow = np.load(base + f"/{fname_root}_flow.npy")
            flow_data = torch.Tensor(
                flow[frame_offset:frame_offset + self.window - 1]).permute(0, 3, 1, 2)  # window-1, c(2), h, w
            return img_data, flow_data, torch.Tensor([label])
        else:
            return img_data, torch.Tensor([label])

    def __len__(self):
        return len(self.clips)


class FightSurvDataset(Dataset):
    def __init__(self, window, flow, train):
        self.root = r'/home/shared/action/Data/fight_surv/'
        train_frac = .85
        self.window = window
        self.data = []
        self.flow = flow
        for clip in glob.glob(self.root + "processed/*"):

            if 'flow' in clip:
                continue
            if "_h5" not in clip:
                continue
            base = clip.split("/")[:-1]
            fname = clip.split("/")[-1]
            fname_root = fname.split('.')[0]
            # print(f"Clip: {fname_root}")
            label = 'fi' == fname_root[:2]  # True if violence
            label = 1 if label else 0
            # Dataset should be about 500GB, 1 node has 256 gb ram. can preload if we find a way to get more memory
            img = h5py.File(clip, 'r')['video']
            # flow = np.load(base+f"/{fname_root}_flow.npy")

            frame_num, w, h, c = img.shape

            self.data.append((frame_num, clip, label))

        self.clips = []
        perm = torch.randperm(len(self.data))
        if train:
            idx = perm[:int(len(self.data)*train_frac)]
        else:
            idx = perm[int(len(self.data)*train_frac):]
        self.data = np.array(self.data)[idx]
        print("Calculating Clips")

        self.clip_agg_data = []

        for video in self.data:
            # print(self.data)
            frame_num, path, label = video
            frame_num = int(frame_num)
            label = int(label)
            clips_so_far = len(self.clips)
            for frame_offset in range(frame_num - window + 1):
                self.clips.append((path, frame_offset, label))
            self.clip_agg_data.append((frame_offset, clips_so_far))

        assert len(self.clip_agg_data) == len(self.data)

    def __getitem__(self, idx):

        total_clips, clips_so_far = self.clip_agg_data[idx]

        # convert idx
        clip_idx = clips_so_far + torch.randint(total_clips, (1,))[0]

        path, frame_offset, label = self.clips[clip_idx]

        img = h5py.File(path, 'r')['video']

        base = self.root + "processed"
        fname_root = path.split("/")[-1].split('.')[0]

        img_data = torch.Tensor(
            img[frame_offset:frame_offset + self.window]).permute(0, 3, 1, 2)  # window, c(3), h, w
        img_data = img_data[:,  [2, 1, 0], :, :]  # fix colors?
        if self.flow:
            flow = np.load(base + f"/{fname_root}_flow.npy")
            flow_data = torch.Tensor(
                flow[frame_offset:frame_offset + self.window - 1]).permute(0, 3, 1, 2)
            return img_data, flow_data, torch.Tensor([label])
        else:
            return img_data, torch.Tensor([label])

    def __len__(self):
        return len(self.data)


class RWFDataset(Dataset):
    def __init__(self, window, train, stride):
        self.root = r'/home/shared/action/Data/RWF-2000-h5/'
        self.window = window
        self.data = []
        self.stride = stride  # if stride ==0, uniformly sample frames across the video

        self.transform = RandomResizedCropCustom((224, 224), scale=(.1, 1.3))

        path = self.root+"train/" if train else self.root+"val/"

        for folder in [path + "Fight/*", path + "NonFight/*"]:
            for clip in glob.glob(folder):

                label = int("NonFight" not in clip)  # True if violence

                img = h5py.File(clip, 'r')['vid_data']

                c, frame_num, w, h = [int(x) for x in img.shape]

                self.data.append((frame_num, clip, label))

    def __getitem__(self, idx):

        frame_num, path, label = self.data[idx]

        img = h5py.File(path, 'r')['vid_data']

        if self.stride > 0:
            n_clips = frame_num - self.window*self.stride+1
            assert n_clips > 0, f"Stride ({self.stride})/Window ({self.window}) too large for this video ({frame_num})"
            starting_clip = torch.randint(n_clips, (1,))[0]
            img_data = torch.Tensor(
                img[:, starting_clip:starting_clip+self.window*self.stride:self.stride])
        else:
            cust_stride = frame_num // self.window
            img_data = torch.Tensor(
                img[:, 0:cust_stride*self.window:cust_stride])

        img_data = img_data.permute(1, 0, 2, 3)  # t, c, h, w
        img_data = self.transform(img_data.unsqueeze(0)).squeeze(0)
        #      img_data = img_data[:,  [2,1,0], :, :]

        return img_data, torch.Tensor([label])

    def __len__(self):
        return len(self.data)


class RWFDataset_npy(Dataset):
    def __init__(self, data_dir, data_partition, clip_len=30, image_size=224, temporal_stride=-1, normalize_mode='video'):
        super(RWFDataset_npy, self).__init__()

        '''
        data_partition = 'train' or 'val'
        clip_len = should be a multiple of 3
        temporal_stride = -1 : 'uniform sampling' // else : 'stride sampling'
        normalize_mode = 'video' or 'imagenet' or 'kinetics'
        '''

        self.data_dir = Path(data_dir)
        self.data_partition = data_partition
        self.video_list = []
        self.labels = []
        self.clip_len = clip_len
        self.temporal_stride = temporal_stride
        self.normalize_mode = normalize_mode

        # Load data list
        i = 0
        for label in ['NonFight', 'Fight']:    
            vid_list = [x for x in self.data_dir.joinpath(label).iterdir() if x.suffix == '.npy']
            self.video_list.extend(vid_list)
            self.labels.extend([i]*len(vid_list))
            i += 1

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        rgb_list = []
        
        video = np.load(self.video_list[idx], mmap_mode='r', allow_pickle=True)
        video = np.float32(video)

        '''stride sampling'''
        if self.data_partition == 'train' and self.temporal_stride != -1:
            data = self.stride_sampling(video, self.clip_len+1, self.temporal_stride)
        else:
            '''uniform sampling'''
            data = self.uniform_sampling(video, self.clip_len+1)

        data[...,:3] = self.color_jitter(data[...,:3])
        data = self.random_flip(data, prob=0.5)

        for image_ in data:
            rgb, _ = np.array_split(image_, 2, axis=-1 )
            rgb_list.append(torch.from_numpy(rgb.transpose(-1,0,1).copy()))
        
        return self.normalize(torch.stack(rgb_list)), self.labels[idx] #[T x C x H x W], scalar

    def random_flip(self, video, prob):
        s = np.random.rand()
        if s < prob:
            video = np.flip(m=video, axis=2)
        return video    

    def stride_sampling(self, video, target_frames, stride):
        vid_len = len(video)

        if vid_len >= (target_frames-1)*stride + 1:
            start_idx = np.random.randint(vid_len - (target_frames-1)*stride)
            data = video[start_idx:start_idx+(target_frames-1)*stride+1:stride]
            

        elif vid_len >= target_frames:
            start_idx = np.random.randint(len(video) - target_frames)
            data = video[start_idx:start_idx + target_frames + 1]

        # Need Zero-pad
        else:
            sampled_video = []
            for i in range(0, vid_len):
                sampled_video.append(video[i])

            num_pad = target_frames - len(sampled_video)
            if num_pad>0:
                while num_pad > 0:
                    if num_pad > len(video):
                        padding = [video[i] for i in range(len(video))]
                        sampled_video += padding
                        num_pad -= len(video)
                    else:
                        padding = [video[i] for i in range(num_pad)]
                        sampled_video += padding
                        num_pad = 0
            data = np.array(sampled_video, dtype=np.float32)
        
        return data
        
    def uniform_sampling(self, video, target_frames):
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames/target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0,len_frames,interval):
            sampled_video.append(video[i])     
        # calculate numer of padded frames and fix it 
        num_pad = target_frames - len(sampled_video)
        if num_pad>0:
            padding = [video[i] for i in range(-num_pad,0)]
            sampled_video += padding     
        # get sampled video
        return np.array(sampled_video, dtype=np.float32)
    
    def color_jitter(self,video):
        # range of s-component: 0-1
        # range of v component: 0-255
        s_jitter = np.random.uniform(-0.2,0.2)
        v_jitter = np.random.uniform(-30,30)
        for i in range(len(video)):
            hsv = cv2.cvtColor(video[i], cv2.COLOR_RGB2HSV)
            s = hsv[...,1] + s_jitter
            v = hsv[...,2] + v_jitter
            s[s<0] = 0
            s[s>1] = 1
            v[v<0] = 0
            v[v>255] = 255
            hsv[...,1] = s
            hsv[...,2] = v
            video[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return video

    def normalize(self, data):
        if self.normalize_mode == 'video':
            mean = torch.mean(data)
            std = torch.std(data)
            return (data-mean) / std
        else:
            mean = torch.FloatTensor(MEAN_STATISTICS[self.normalize_mode])
            std = torch.FloatTensor(STD_STATISTICS[self.normalize_mode])
            return (data/255.-mean.view(3,1,1)) / std.view(3,1,1)            

