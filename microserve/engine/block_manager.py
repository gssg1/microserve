from .sequence import Sequence
from collections import deque
import xxhash
import numpy as np

class Block:
    def __init__(self,block_id: int):
        self.Block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.num_tokens = 0

    def update(self,hash: int,token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1
        self.num_tokens = 0
        self.token_ids = []
        self.ref_count = 1

class BlockManager:
    def __init__(self,nums_block: int,block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(id) for id in range(nums_block)]
        self.hash_to_block: dict[int,int] = dict()
        self.free_blocks: deque[int] = deque(range(nums_block))
        self.used_blocks: set[int] = set()

    @property
    def free_blocks_nums(self) -> int:
        return len(self.free_blocks)
    
    @classmethod
    def compute_hash(cls,token_ids: list[int],prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8,'little'))
        h.update(np.asarray(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    def allocate_block(self) -> int:
        block_id = self.free_blocks.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block.get(block.hash) == block_id:
            del self.hash_to_block[block.hash]
        block.reset()
        self.used_blocks.add(block_id)
        return block_id

    def deallocate_block(self,block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.used_blocks.remove(block_id)
        self.free_blocks.append(block_id)

    def allocate_cache(self,seq: Sequence) -> list[int]:
        nums_block = seq.num_tokens_with_spec // self.block_size
        hash = -1
        block_table = []
        for i in range(nums_block - 1):
            hash = self.compute_hash(seq.token_ids[i*self.block_size:(i+1)*self.block_size],hash)
            block_id = self.hash_to_block.get(hash,-1)
            if (block_id != -1 and
                    self.blocks[block_id].token_ids ==
                    seq.token_ids[i*self.block_size:(i+1)*self.block_size]):
                block_table.append(block_id)
                self.blocks[block_id].ref_count += 1
            else:
                break
        return block_table

    def allocate(self,seq: Sequence,token_budget: int) -> bool:
        if (seq.gap == 1):
            if (seq.num_computed_tokens % self.block_size != 0):
                return True
            else:
                if (self.free_blocks_nums > 0):
                    block_id = self.allocate_block()
                    seq.block_table.append(block_id)
                    return True
                else:
                    return False
        else:
            if (seq.num_computed_tokens == 0):
                block_table = self.allocate_cache(seq)
                for i in range(len(block_table)):
                    block_id = block_table[i]
                    seq.block_table.append(block_id)

                    seq.num_computed_tokens += self.block_size
                    if (block_id not in self.used_blocks):
                        self.used_blocks.add(block_id)
                        self.free_blocks.remove(block_id)

                num_scheduled_tokens = min(self.free_blocks_nums * self.block_size,token_budget,seq.gap)
                if (num_scheduled_tokens == 0 and seq.gap > 0):
                    for block_id in block_table:
                        self.blocks[block_id].ref_count -= 1
                        if self.blocks[block_id].ref_count == 0:
                            self.deallocate_block(block_id)
                    del seq.block_table[:len(block_table)]
                    seq.num_computed_tokens -= len(block_table) * self.block_size
                    return False
                seq.num_scheduled_tokens = num_scheduled_tokens

                for i in range((num_scheduled_tokens - 1) // self.block_size + 1):
                    block_id = self.allocate_block()
                    seq.block_table.append(block_id)
            else:
                last_used = seq.num_computed_tokens % self.block_size
                last_free = 0 if last_used == 0 else self.block_size - last_used
                num_scheduled_tokens = min(self.free_blocks_nums * self.block_size + last_free,
                                           token_budget, seq.gap)
                seq.num_scheduled_tokens = num_scheduled_tokens

                num_new_blocks = (max(0, num_scheduled_tokens - last_free)
                                  + self.block_size - 1) // self.block_size
                for i in range(num_new_blocks):
                    block_id = self.allocate_block()
                    seq.block_table.append(block_id)
            return True

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.deallocate_block(block_id)
        seq.block_table.clear()

    def hash_blocks(self, seq: Sequence):
        start = seq.num_computed_tokens // self.block_size
        end = (seq.num_computed_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.token_ids[i*self.block_size:(i+1)*self.block_size]
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block[h] = block.Block_id

            
                    

                
