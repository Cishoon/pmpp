from model import model, input_size
from time import time
import torch    

DEVICE = "cuda:0"

@torch.no_grad()
def main(num_forward_passes: int = 2000): 
    model.to(DEVICE)

    x = torch.randn(1, input_size)
    x = x.to(DEVICE)

    for _ in range(num_forward_passes):
        output = model(x)


if __name__ == '__main__':

    

    start = time()
    main()
    stop = time()

    print(f"Duration for 1 GPU setting: {stop - start}")

