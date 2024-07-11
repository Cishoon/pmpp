from model import model, input_size
from time import time, sleep
import torch
import torch.distributed as dist
from torch import nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


@torch.no_grad()
def main(num_forward_passes: int = 2000): 
    x = torch.randn(1, input_size)

    for _ in range(num_forward_passes):
        output = model(x)


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    tp_mesh = init_device_mesh("cuda", (torch.cuda.device_count(),))

    tp_plan = {}
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, nn.Linear):
            tp_plan[f"layers.{idx}"] = ColwiseParallel() if idx % 2 == 0 else RowwiseParallel()

    model = parallelize_module(model, tp_mesh, tp_plan)

    start = time()
    main()
    torch.cuda.synchronize() 
    stop = time()

    if local_rank == 0:
        print(f"Duration for {torch.cuda.device_count()} GPUs setting: {stop - start}")

    dist.destroy_process_group()

    sleep(5)

