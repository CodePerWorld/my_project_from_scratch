import torch

if __name__ == '__main__':
    a = torch.Tensor([
        [[1,2,3],[1,2,3]],
        [[1,2,3],[1,2,3]]
    ])
    print(a.shape)
    mask = torch.tensor([1,1,0])
    a = a.masked_fill(mask==0, float("-1e20"))
    print(a)