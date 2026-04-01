import os

import torch
try:
    #deps for cambricon devices
    import torch_mlu
    from torch_mlu.utils.gpu_migration import migration
    dev_name = "mlu"
except:
    try:
        import torch_npu
        from torch_npu.contrib import transfer_to_npu
        dev_name = "npu"
    except:
        try:
            import torch_musa
            import transfer_to_musa
            dev_name = "musa"
        except:
            try:
                import torch_txda
                from torch_txda import transfer_to_txda
                dev_name = "txda"
            except:
                try:
                    import torch_gcu
                    from torch_gcu import transfer_to_gcu
                    dev_name = "gcu"
                except:
                    dev_name = "cuda"

import sdccl
import torch.distributed as dist
import argparse
#import pdb
#pdb.set_trace()

SDCCL_GROUP1 = None
SDCCL_GROUP2 = None
SDCCL_GROUP3 = None
MY_RANK = None
WORLD_SIZE = None
PREV_RANK = None
NEXT_RANK = None

def get_args():
    parser = argparse.ArgumentParser(description='SDCCL PyTorch Plugin ArgumentParser')
    parser.add_argument('-o', '--op',
                        default='all',
                        choices=['broadcast',
                                 'reduce',
                                 'allreduce',
                                 'allgather',
                                 'reducescatter',
                                 'sendrecv',
                                 'gather',
                                 'scatter',
                                 'alltoall',
                                 'all'],
                        help='operation to test (default: all)')
    return parser.parse_args()

def init_pg():
    global SDCCL_GROUP1, SDCCL_GROUP2, SDCCL_GROUP3, MY_RANK, WORLD_SIZE, PREV_RANK, NEXT_RANK

    # Get rank and world_size from environment
    MY_RANK = int(os.environ["RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize the default sdccl process group
    dist.init_process_group(f"cpu:gloo,{dev_name}:sdccl", rank=MY_RANK, world_size=WORLD_SIZE)
    print(f"ddp backend config is {dist.get_backend_config()}")

    # Create two groups
    ranks = list(range(WORLD_SIZE))
    SDCCL_GROUP1 = dist.new_group(ranks=ranks, backend=f"cpu:gloo,{dev_name}:sdccl")
    SDCCL_GROUP2 = dist.new_group(ranks=ranks, backend=f"cpu:gloo,{dev_name}:sdccl")
    print(f"ranks_sdccl: {dist.get_process_group_ranks(SDCCL_GROUP1)}")

    # Create a group with options; this only works when sdcclBackend has Options defined
    # TODO: confirm with all vendors to see if their torch implementation support backend options
    if sdccl._C is not None and hasattr(sdccl._C, 'ProcessGroupSDCCL') and hasattr(sdccl._C.ProcessGroupSDCCL, 'Options'):
        sdccl_options = sdccl._C.ProcessGroupSDCCL.Options(enable_tuner=True)
        SDCCL_GROUP3 = dist.new_group(ranks=ranks, backend=f"{dev_name}:sdccl", pg_options=sdccl_options)
        print(f"ranks_sdccl with options: {dist.get_process_group_ranks(SDCCL_GROUP3)}")

    # Get prev_rank and next_rank
    PREV_RANK = (MY_RANK - 1 + WORLD_SIZE) % WORLD_SIZE
    NEXT_RANK = (MY_RANK + 1) % WORLD_SIZE
    print(f"rank: {MY_RANK}, world_size = {WORLD_SIZE}, prev_rank: {PREV_RANK}, next_rank: {NEXT_RANK}")

    if torch.cuda.is_available():
        # Set device
        torch.cuda.set_device(local_rank)

def destroy_pg():
    dist.destroy_process_group()
    

def test_broadcast():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()

        # Perform broadcast with SDCCL_GROUP2
        print(f"rank {MY_RANK} before broadcast with SDCCL_GROUP2: x = {x}")
        dist.broadcast(x, 0, group=SDCCL_GROUP2)
        print(f"rank {MY_RANK} after broadcast with SDCCL_GROUP2: x = {x}")

def test_reduce():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        for i in range(WORLD_SIZE):
            x[i] = i

        # Perform reduce with SDCCL_GROUP1
        print(f"rank {MY_RANK} before reduce on src rank 0 with SDCCL_GROUP1: x = {x}")
        dist.reduce(x, 0, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after reduce on src rank 0 with SDCCL_GROUP1: x = {x}")

def test_allreduce():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        y = torch.rand(WORLD_SIZE).cuda()
        print(f"rank {MY_RANK} before allreduce: x = {x}, y = {y}")
        # Perform allreduce with SDCCL_GROUP1
        dist.all_reduce(x, op=dist.ReduceOp.MIN, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after allreduce min with SDCCL_GROUP1: x = {x}")
        dist.all_reduce(y, op=dist.ReduceOp.MAX, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after allreduce max with SDCCL_GROUP1: y = {y}")
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after allreduce sum with SDCCL_GROUP1: x = {x}")

        # Perform all_reduce_coalesced with SDCCL_GROUP1
        print(f"rank {MY_RANK} before all_reduce_coalesced sync sum with SDCCL_GROUP1: x = {x}, y = {y}")
        with dist._coalescing_manager(group=SDCCL_GROUP1):
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
            dist.all_reduce(y, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after all_reduce_coalesced sync sum with SDCCL_GROUP1: x = {x}, y = {y}")
        print(f"rank {MY_RANK} before all_reduce_coalesced async sum with SDCCL_GROUP1: x = {x}, y = {y}")
        with dist._coalescing_manager(group=SDCCL_GROUP1, async_ops=True)  as cm:
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
            dist.all_reduce(y, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
        cm.wait()
        print(f"rank {MY_RANK} after all_reduce_coalesced async sum with SDCCL_GROUP1: x = {x}, y = {y}")

def test_sendrecv():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        y = torch.rand(WORLD_SIZE).cuda()
        for i in range(WORLD_SIZE):
            x[i] = MY_RANK

        # Perform send and recv with SDCCL_GROUP2
        print(f"rank {MY_RANK} before batch_isend_irecv with SDCCL_GROUP2: x = {x}, y = {y}")
        op_send = dist.P2POp(dist.isend, x, NEXT_RANK, group=SDCCL_GROUP2)
        op_recv = dist.P2POp(dist.irecv, y, PREV_RANK, group=SDCCL_GROUP2)
        op_list = [op_send, op_recv]
        reqs = dist.batch_isend_irecv(op_list)
        for req in reqs:
            req.wait()
        print(f"rank {MY_RANK} after batch_isend_irecv with SDCCL_GROUP2: x = {x}, y = {y}")
        if MY_RANK % 2 == 0:
            dist.send(x, NEXT_RANK, group=SDCCL_GROUP2)
            dist.recv(y, PREV_RANK, group=SDCCL_GROUP2)
        elif MY_RANK % 2 == 1:
            dist.recv(y, PREV_RANK, group=SDCCL_GROUP2)
            dist.send(x, NEXT_RANK, group=SDCCL_GROUP2)
        handle = dist.barrier(group=SDCCL_GROUP2, async_op=True)
        handle.wait()
        print(f"rank {MY_RANK} after send/recv with SDCCL_GROUP2: x = {x}, y = {y}")

def test_allgather():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        y = torch.rand(WORLD_SIZE).cuda()
        z = torch.rand(1).cuda()
        z[0] = MY_RANK

        # Perform allgather with SDCCL_GROUP1
        print(f"rank {MY_RANK} before _all_gather_base with SDCCL_GROUP1: z = {z}, y = {y}")
        dist._all_gather_base(y, z, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after _all_gather_base with SDCCL_GROUP1: z = {z}, y = {y}")
        z_list = list(torch.chunk(x, WORLD_SIZE, dim=0))
        print(f"rank {MY_RANK} before all_gather with SDCCL_GROUP1: z = {z}, z_list = {z_list}")
        dist.all_gather(z_list, z, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after all_gather with SDCCL_GROUP1: z = {z}, z_list = {z_list}")
        all_rank_infos = [None] * WORLD_SIZE
        #cur_rank_info = {'rank': MY_RANK, 'device_type': f"cpu:gloo,{dev_name}:sdccl"}
        cur_rank_info = [MY_RANK, MY_RANK+1]
        dist.all_gather_object(all_rank_infos, cur_rank_info, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after all_gather_object with SDCCL_GROUP1: all_rank_infos = {all_rank_infos}")

        # Perform all_gather_coalesced with SDCCL_GROUP1
        z1 = torch.rand(1).cuda()
        z1[0] = MY_RANK * 10
        print(f"rank {MY_RANK} before all_gather_coalesced sync with SDCCL_GROUP1: x = {x}, y = {y}, z = {z}, z1 = {z1}")
        with dist._coalescing_manager(group=SDCCL_GROUP1):
            dist.all_gather_into_tensor(x, z1, group=SDCCL_GROUP1)
            dist.all_gather_into_tensor(y, z, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after all_gather_coalesced sync with SDCCL_GROUP1: x = {x}, y = {y}, z = {z}, z1 = {z1}")
        x = torch.rand(WORLD_SIZE).cuda()
        y = torch.rand(WORLD_SIZE).cuda()
        print(f"rank {MY_RANK} before all_gather_coalesced async with SDCCL_GROUP1: x = {x}, y = {y}, z = {z}, z1 = {z1}")
        with dist._coalescing_manager(group=SDCCL_GROUP1, async_ops=True)  as cm:
            dist.all_gather_into_tensor(x, z1, group=SDCCL_GROUP1)
            dist.all_gather_into_tensor(y, z, group=SDCCL_GROUP1)
        cm.wait()
        print(f"rank {MY_RANK} after all_gather_coalesced async with SDCCL_GROUP1: x = {x}, y = {y}, z = {z}, z1 = {z1}")

def test_reducescatter():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        y = torch.rand(WORLD_SIZE).cuda()
        z = torch.rand(1).cuda()
        z1 = torch.rand(1).cuda()
        z[0] = 0
        z1[0] = MY_RANK * 10

        # Perform reducescatter with SDCCL_GROUP1
        for i in range(WORLD_SIZE):
            x[i] = i
        x_list = list(torch.chunk(x, WORLD_SIZE, dim=0))
        print(f"rank {MY_RANK} before reduce_scatter with SDCCL_GROUP1: x_list = {x_list}, z = {z}")
        dist.reduce_scatter(z, x_list, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after reduce_scatter with SDCCL_GROUP1: x_list = {x_list}, z = {z}")
        for i in range(WORLD_SIZE):
            x[i] = MY_RANK
        print(f"rank {MY_RANK} before _reduce_scatter_base with SDCCL_GROUP1: x = {x}, z = {z}")
        dist._reduce_scatter_base(z, x, op=dist.ReduceOp.MAX, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after _reduce_scatter_base with SDCCL_GROUP1: x = {x}, z = {z}")

        # Perform reduce_scatter_coalesced with SDCCL_GROUP1
        x = torch.rand(WORLD_SIZE).cuda()
        y = torch.rand(WORLD_SIZE).cuda()
        z[0] = 0
        z1[0] = 0
        print(f"rank {MY_RANK} before reduce_scatter_coalesced sync with SDCCL_GROUP1: x = {x}, y = {y}, z={z}, z1={z1}")
        with dist._coalescing_manager(group=SDCCL_GROUP1):
            dist.reduce_scatter_tensor(z, x, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
            dist.reduce_scatter_tensor(z1, y, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after reduce_scatter_coalesced sync with SDCCL_GROUP1: x = {x}, y = {y}, z={z}, z1={z1}")
        z[0] = 0
        z1[0] = 0
        print(f"rank {MY_RANK} before reduce_scatter_coalesced async with SDCCL_GROUP1: x = {x}, y = {y}, z={z}, z1={z1}")
        with dist._coalescing_manager(group=SDCCL_GROUP1, async_ops=True)  as cm:
            dist.reduce_scatter_tensor(z, x, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
            dist.reduce_scatter_tensor(z1, y, op=dist.ReduceOp.SUM, group=SDCCL_GROUP1)
        cm.wait()
        print(f"rank {MY_RANK} after reduce_scatter_coalesced async with SDCCL_GROUP1: x = {x}, y = {y}, z={z}, z1={z1}")


def test_gather():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        x_list = list(torch.chunk(x, WORLD_SIZE, dim=0))
        z = torch.rand(1).cuda()
        z[0] = MY_RANK

        # Perform gather with SDCCL_GROUP1
        print(f"rank {MY_RANK} before gather on dst rank 0 with SDCCL_GROUP1: z = {z}, x_list = {x_list}")
        if MY_RANK == 0:
            handle = dist.gather(z, x_list, 0, group=SDCCL_GROUP1, async_op=True)
        else:
            handle = dist.gather(z, None, 0, group=SDCCL_GROUP1, async_op=True)
        handle.wait()
        print(f"rank {MY_RANK} after gather on dst rank 0 with SDCCL_GROUP1: z = {z}, x_list = {x_list}")

def test_scatter():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        x_list = list(torch.chunk(x, WORLD_SIZE, dim=0))
        z = torch.rand(1).cuda()
        z[0] = -1

        # Perform scatter with SDCCL_GROUP2
        print(f"rank {MY_RANK} before scatter from src rank 0 with SDCCL_GROUP2: z = {z}, x_list = {x_list}")
        if MY_RANK == 0:
            dist.scatter(z, x_list, 0, group=SDCCL_GROUP2)
        else:
            dist.scatter(z, None, 0, group=SDCCL_GROUP2)
        print(f"rank {MY_RANK} after scatter from src rank 0 with SDCCL_GROUP2: z = {z}, x_list = {x_list}")

def test_alltoall():
    if torch.cuda.is_available():
        # Create tensors
        x = torch.rand(WORLD_SIZE).cuda()
        y = torch.rand(WORLD_SIZE).cuda()
        for i in range(WORLD_SIZE):
            x[i] = MY_RANK
            y[i] = 0
        list_x = list(torch.chunk(x, WORLD_SIZE, dim=0))
        list_y = list(torch.chunk(y, WORLD_SIZE, dim=0))

        # Perform all_to_all with SDCCL_GROUP1
        print(f"rank {MY_RANK} before all_to_all with SDCCL_GROUP1: list_x = {list_x}, list_y = {list_y}")
        dist.all_to_all(list_y, list_x, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after all_to_all with SDCCL_GROUP1: list_x = {list_x}, list_y = {list_y}")

        # Perform all_to_all_single with SDCCL_GROUP2
        ## Without output_splits, input_splits
        for i in range(WORLD_SIZE):
            x[i] = MY_RANK
            y[i] = 0
        print(f"rank {MY_RANK} before all_to_all_single with SDCCL_GROUP1 (no splits): x = {x}, y = {y}")
        handle = dist.all_to_all_single(y, x, group=SDCCL_GROUP1, async_op=True)
        handle.wait()
        print(f"rank {MY_RANK} after all_to_all_single with SDCCL_GROUP1 (no splits): x = {x}, y = {y}")

        ## With output_splits, input_splits
        input_splits = []
        output_splits = []
        for i in range(WORLD_SIZE):
            x[i] = MY_RANK
            y[i] = 0
            if MY_RANK % 2 == 0:
                if i % 2 == 0:
                    input_splits.append(2)
                    output_splits.append(2)
                else:
                    input_splits.append(0)
                    output_splits.append(0)
            else:
                if i % 2 == 1:
                    input_splits.append(2)
                    output_splits.append(2)
                else:
                    input_splits.append(0)
                    output_splits.append(0)
        print(f"rank {MY_RANK} before all_to_all_single with SDCCL_GROUP1 (with splits): x = {x}, y = {y}")
        dist.all_to_all_single(y, x, output_splits, input_splits, group=SDCCL_GROUP1)
        print(f"rank {MY_RANK} after all_to_all_single with SDCCL_GROUP1 (with splits): x = {x}, y = {y}")

def test_all():
    test_broadcast()
    test_reduce()
    test_allreduce()
    test_sendrecv()
    test_allgather()
    test_reducescatter()
    test_gather()
    test_scatter()
    test_alltoall()

dict_op_to_test = {
    "broadcast": test_broadcast,
    "reduce": test_reduce,
    "allreduce": test_allreduce,
    "allgather": test_allgather,
    "reducescatter": test_reducescatter,
    "sendrecv": test_sendrecv,
    "gather": test_gather,
    "scatter": test_scatter,
    "alltoall": test_alltoall,
    "all": test_all
}

if __name__ == "__main__":
    init_pg()

    args = get_args()

    dict_op_to_test.get(args.op, test_all)()

    destroy_pg()
    
