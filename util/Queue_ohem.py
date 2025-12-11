# File: util/Queue_ohem.py

import torch
import torch.nn as nn

class OHEMQueue(nn.Module):
    """
    A queue implementation using a PyTorch tensor buffer, designed for on-device (GPU) operations.
    This queue is specifically for the Light-OHEM strategy and includes storage for difficulty scores.
    """
    def __init__(self, max_size, embedding_dim):
        super(OHEMQueue, self).__init__()
        self.max_size = max_size
        self.embedding_dim = embedding_dim
        
        # register_buffer ensures the tensor is part of the module's state and moves with .to(device)
        self.register_buffer('embeddings', torch.zeros(self.max_size, self.embedding_dim))
        self.register_buffer('difficulty_scores', torch.zeros(self.max_size))
        
        self.ptr = 0
        self.is_full_flag = False

    @torch.no_grad()
    def enqueue(self, embeddings_batch):
        """
        Adds a batch of embeddings to the queue.
        Embeddings should already be on the correct device.
        """
        batch_size = embeddings_batch.shape[0]
        if batch_size == 0:
            return
            
        # Determine the indices to write to
        ptr_end = self.ptr + batch_size
        if ptr_end <= self.max_size:
            # The batch fits without wrapping around
            indices = torch.arange(self.ptr, ptr_end)
            self.embeddings[indices] = embeddings_batch
            self.ptr = ptr_end % self.max_size
        else:
            # The batch wraps around the end of the buffer
            num_to_end = self.max_size - self.ptr
            self.embeddings[self.ptr:] = embeddings_batch[:num_to_end]
            
            num_remaining = batch_size - num_to_end
            self.embeddings[:num_remaining] = embeddings_batch[num_to_end:]
            self.ptr = num_remaining

        if not self.is_full_flag and self.ptr >= self.max_size - batch_size:
            self.is_full_flag = True

    def get_all_embeddings(self):
        """Returns all valid embeddings currently in the queue."""
        if not self.is_full_flag:
            # If not full, only return the part that has been filled
            return self.embeddings[:self.ptr]
        else:
            return self.embeddings

    def size(self):
        """Returns the current number of elements in the queue."""
        return self.max_size if self.is_full_flag else self.ptr

    def is_full(self):
        return self.is_full_flag