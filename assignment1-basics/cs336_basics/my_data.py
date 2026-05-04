import numpy as np
import torch,os
from typing import BinaryIO, IO
import numpy.typing as npt

def data_loading(data_set: npt.NDArray[int],
                 batch_size: int,
                 context_length: int,
                 device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, data_set.shape[0]-context_length, size=batch_size)
    idx = starts[:,None]+np.arange(context_length)
    return torch.from_numpy(data_set[idx]).to(device=device, dtype=torch.int32), \
                torch.from_numpy(data_set[idx+1]).to(device=device, dtype=torch.int32)
    
def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def run_load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                        model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']