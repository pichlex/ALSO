import torch
import torch.distributed as dist
import os


def ddp_max_oracle(max_oracle, losses, src_device=0):
    """Take any existing maximization oracle and apply it to multiple devices using a gather-scatter implementation
    within the data distributed parallel (DDP) framework. Assumes that process rank is discoverable, e.g. the job is run using `torchrun`.
    
    Args:
      max_oracle: a function that consumes ``n`` (full-batch size) loss values and returns ``n`` weights (where ``n == micro_size * n_gpus``)
      losses: a PyTorch tensor of ``micro_size`` losses
    
    Returns:
      weights
        a vector of weights of size ``len(losses)`` indicating the weight on each example
    """

    # type check
    assert callable(max_oracle), f"`max_oracle` must be a function, got {type(max_oracle)} instead"
    assert torch.is_tensor(losses), f"`losses` must by a PyTorch tensor, got {type(losses)} instead"

    # discover the index of this process
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    micro_size = len(losses)

    # quantilize the current micro-batch of losses
    sort, argsort = torch.sort(losses, stable=True)

    # allocate memory for full vector of losses then gather
    gather_list = [torch.zeros(micro_size).to(src_device) for _ in range(world_size)] if rank == 0 else None
    dist.gather(sort, gather_list=gather_list, dst=src_device)

    # if source rank, then apply oracle to gathered list
    q_batch = torch.zeros(micro_size).to(f"cuda:{rank}")
    if rank == 0:
        q = max_oracle(torch.cat(gather_list))
        scatter_list = [q[micro_size * r: micro_size * (r + 1)].to(src_device) for r in range(world_size)]
        dist.scatter(q_batch, scatter_list=scatter_list, src=src_device)
    else:
        dist.scatter(q_batch, src=src_device)

    # unsort to assign weights to data indices
    return q_batch[torch.argsort(argsort)]
