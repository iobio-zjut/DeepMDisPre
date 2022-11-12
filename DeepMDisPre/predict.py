import sys
import numpy as np
import torch
import torch.nn.functional as F
from Network import Network
from esm_predict import msa_trans

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 指定编号为0的一张显卡，"0,1,2,3"是4张

a3m_path = sys.argv[1]
feat47_path = sys.argv[2]
msa_128_path = sys.argv[3]
out = sys.argv[4]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create neural network model (depending on first command line parameter)
    model = Network().eval().to(device)

    pretrained_dict = torch.load('./pt-model/model.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained_dict)

    # sequence length
    with open(a3m_path, 'r') as fa:
        L = fa.readline().strip().__len__()

    # input features
    feat47 = np.memmap(feat47_path, dtype=np.float32, mode='r', shape=(1, 47, L, L))
    feat47 = torch.from_numpy(feat47).type(torch.FloatTensor).permute(0, 2, 3, 1).to(device)

    msa_feats, row_att = msa_trans(msa_128_path)
    msa_feats = msa_feats.to(device)
    row_att = row_att.to(device)

    with torch.no_grad():
        pred_dis = model(feat47, msa_feats, row_att)
        dist = F.softmax(pred_dis, dim=1)

        dist = torch.squeeze(dist).permute(1, 2, 0).cpu()

        dist = dist.numpy()

        np.savez_compressed(out, dist=dist)


if __name__ == "__main__":
    main()
