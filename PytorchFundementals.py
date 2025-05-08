import sys
import numpy as np
import torch

def main():
    t1 = torch.zeros((4, 2, 3))
    print(t1.shape)  # try t1.shape[0], t1.shape[1], t1.shape[2], t1.shape[-1]

    # To add an extra dimension in the beginning i.e., to make the shape(1,4,2,3)
    t2 = t1.reshape(-1, 4, 2, 3)
    print(t2.shape)

    t3 = t1.view(-1, 4, 2, 3)
    print(t3.shape)

    # To add a dimension at the end, you can use view or reshape like before
    t4 = t1.view(4, 2, 3, -1)
    print(t4.shape)

    # To add a dimension in the beginning or end, you can also use unsqueeze
    t5 = t1.unsqueeze(dim=0)
    print(t5.shape)

    # you can squeeze to remove the dimension
    t6 = t5.squeeze(dim=0)
    print(t6.shape)

    t7 = t1.reshape(4, 6)
    print('t7', t7.shape)

    t8 = t1.reshape(4, -1)
    print('t8', t8.shape)

    # * to unpack a tuple or a list, zip operation to combine two tuples or lists
    aa = [(2, 3, 5), (6, 7, 8)]
    print('unpacked list:', *aa)
    bb, cc, dd = zip(*aa)
    print('bb:', bb)

    a = np.array([[5, 3], [6, 7]])
    b = torch.tensor(a, dtype=torch.int64)[None]
    print('After None shape:', b.shape)

    # gather allows us to select some of the elements from a tensor
    t = torch.tensor([[1, 2], [3, 4]])
    r = torch.gather(t, dim=1, index=torch.tensor([[1, 0], [0, 1]]))
    print('r:', r)

    w = t.view(-1)
    print('w:', w)

    v = t.view(-1)[:, None]
    print('v:', v.shape)

    r2 = torch.gather(t, dim=0, index=torch.tensor([[1, 0], [0, 1]]))
    print('r2:', r2)

    r3 = torch.gather(t, dim=0, index=torch.tensor([[1], [0]]))
    print('r3:', r3)

    out1 = torch.tensor([
        [0.10, 0.50, 0.40],
        [0.55, 0.20, 0.25],
        [0.60, 0.10, 0.30],
        [0.15, 0.65, 0.20]
    ])
    print('out1:', out1.shape)

    y = torch.tensor([1, 2, 0, 1], dtype=torch.int64).reshape(4, 1)
    probs = out1.gather(dim=1, index=y)
    print(probs)

    y2 = torch.tensor([[1, 1], [2, 0], [0, 1], [1, 2]], dtype=torch.int64)
    probs2 = out1.gather(dim=1, index=y2)
    print(probs2)

    out1_mean = out1.mean(dim=1, keepdim=True)
    print(out1_mean)

    state = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    mask = ~(state != 0) * 1
    mask2 = (state != 0) * -1000
    mask3 = (state > 0).float()
    print(mask3)

    board = [0, 0, 1, 0, 2, 0, 1, 2, 0]
    print(np.eye(3))
    print(np.eye(3)[board])
    print('---------')
    print(np.eye(3)[board][:, [0, 2, 1]])

if __name__ == "__main__":
    main()
