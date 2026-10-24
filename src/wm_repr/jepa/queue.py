import torch
import torch.nn as nn


class KeyQueue(nn.Module):
    """
    FIFO key queue with three tracked buffers:
      - queue: (K, D) tensor
      - queue_ptr: scalar long pointing to next write index (0..K-1)
      - length: scalar long number of valid items currently in queue (0..K)
    """

    def __init__(self, queue_size: int, embedding_dim: int):
        super().__init__()
        self.K = queue_size
        self.D = embedding_dim
        self.register_buffer("queue", torch.zeros(self.K, self.D), persistent=True)
        self.register_buffer("queue_ptr", torch.tensor(0, dtype=torch.long), True)
        self.register_buffer("length", torch.tensor(0, dtype=torch.long), True)

    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor):
        """
        Enqueue keys: keys (B, D), stored as given.
        - If B >= K, keep only last K keys (newest K).
        - Update queue_ptr and length (length increases up to K).
        """
        B, D = keys.shape
        assert D == self.D, f"Key dimension {D} does not match queue dimension {self.D}"

        K = self.K
        if B >= K:
            # keep last K entries (newest)
            self.queue.copy_(keys[-K:])
            self.queue_ptr.fill_(0)
            self.length.fill_(K)
            return

        ptr = int(self.queue_ptr.item())  # next write index
        end = ptr + B
        if end <= K:
            # contiguous write
            self.queue[ptr:end].copy_(keys)
        else:
            # wrap-around: write tail, then head
            first_part = K - ptr
            self.queue[ptr:].copy_(keys[:first_part])
            second_part = B - first_part
            self.queue[:second_part].copy_(keys[first_part:])

        # advance pointer circularly
        new_ptr = (ptr + B) % K
        self.queue_ptr.fill_(new_ptr)

        # update filled length (increase up to K)
        new_length = min(int(self.length.item()) + B, K)
        self.length.fill_(new_length)

    def get(self) -> torch.Tensor:
        """
        Return chronological slice of valid entries (oldest -> newest), shape (N, D),
        where N == length. Returns empty tensor if length == 0 or queue uninitialized.
        """
        if self.queue.numel() == 0 or int(self.length.item()) == 0:
            return torch.empty(0)

        n = int(self.length.item())
        ptr = int(self.queue_ptr.item())
        K = self.queue.shape[0]

        # If buffer full (n == K) return full queue starting from ptr (oldest at ptr)
        if n == K:
            # chronological order oldest->newest: queue[ptr:] + queue[:ptr]
            if ptr == 0:
                return self.queue.clone().detach()
            return torch.cat(
                [self.queue[ptr:].clone(), self.queue[:ptr].clone()], dim=0
            ).detach()

        # The queue fills from index 0 and has not wrapped yet, so the valid
        # entries are queue[0:n].
        assert (ptr - n) % K == 0
        return self.queue[:n].clone().detach()
