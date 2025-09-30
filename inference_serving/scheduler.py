import pandas as pd
from time import time
import csv

from .request import *
from .utils import *
from .control import *
from .memory_model import *
from .generate_graph import *
from .generate_trace import *
from .pim import *
from .data_loader import create_data_loader

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, max_batch, npu_num, npu_group, npu_mem, fp, block_size, req_num, verbose=False):
        # all time realated variables are in using tick (system tick)
        # LLMServingSim uses Orca, vLLM technique at deafult
        self.model = model
        self.max_batch = max_batch
        self.npu_num = npu_num
        self.npu_group = npu_group
        self.req_num = req_num
        # lists are sorted in arrival time manner
        self.request = [] # list of requests
        self.inflight = [] # list of batches
        self.done = [] # list of requests
        self.req_ids = -1
        self.batch_ids = -1

        # memory model
        self.memory = MemoryModel(model, npu_num, npu_mem, block_size, fp, verbose)

        # universal data loader
        self.data_loader = create_data_loader(verbose)

        # verbose
        self.verbose = verbose

    # generate request from dataset
    def generate(self, path, is_init=True):
        path = f'../{path}' # move out from astra-sim folder

        # Use universal data loader
        try:
            data = self.data_loader.load_dataset(path, self.req_num)
        except Exception as e:
            print(f"Scheduler: Error loading dataset {path}: {e}")
            return

        cnt = 0
        for index, row in data.iterrows():
            if index >= self.req_num:
                break

            input_length = int(row['input_toks'])
            output_length = int(row['input_toks'] + row['output_toks'])
            arrival_time_ns = int(row['arrival_time_ns'])

            # Add additional metadata if available
            request_data = [self.model, input_length, output_length, arrival_time_ns]

            self.add_request(request_data, is_init=is_init)
            cnt += 1

        if self.verbose:
            print(f"Scheduler: added {cnt} requests to LLMServingSim from {path}")

            # Show dataset statistics if available
            if 'model_type' in data.columns:
                model_dist = data['model_type'].value_counts()
                print(f"Scheduler: Model distribution: {dict(model_dist)}")

            if 'burst_pattern' in data.columns:
                burst_dist = data['burst_pattern'].value_counts()
                print(f"Scheduler: Burst pattern distribution: {dict(burst_dist)}")

        return

    # batch the request scheduling method
    def schedule(self, current, sys, batch_id=-1):
        # first NPU to process new batch
        if sys == 0:
            # nothing to batch return None
            if len(self.request) != 0 and self.request[0].arrival > current:
                return None
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.npu_group:
                # wait it to be done
                return None

            # scheduling start
            batch_req = [req for req in self.request if req.arrival <= current]
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch

            # nothing to batch
            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            kv_size = 0
            evict_size = 0
            gen_req = [req for req in batch_req if not req.is_init]
            # check if there is request that need to enlarge the block
            temp_len = batch_len
            for i in range(batch_len, -1, -1):
                kv_size = self.memory.get_block_kv(batch_req, i) # includes evicted input, and initiation input
                if self.memory.mem_avail(kv_size):
                    temp_len = i
                    break
            
            # no memory to batch
            while temp_len == 0:
                # preempt request one by one untill there is enough space
                if len(gen_req) == 0:
                    return None
                
                # check already evicted request
                if gen_req[-1].evict:
                    gen_req = gen_req[:-1]
                    continue

                # else
                evict_size = self.memory.get_evict_kv(gen_req[-1])
                gen_req[-1].evict = True
                if self.verbose:
                    print(f"Sceduler: eviction of the request #{gen_req[-1].id}")
                gen_req = gen_req[:-1]
                self.memory.mem_store(evict_size)

                if len(gen_req) < batch_len:
                    batch_len = len(gen_req)

                # check if can batch
                for i in range(batch_len, -1, -1):
                    kv_size = self.memory.get_block_kv(batch_req, i)
                    if self.memory.mem_avail(kv_size):
                        temp_len = i
                        break

            batch_len = temp_len
            batch_req = batch_req[:batch_len]
            load_size = 0

            # delete from request queue
            for req in batch_req:
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break

                if req.evict:
                    # load evicted kv cache
                    load_size += self.memory.get_evict_kv(req)
                    req.evict = False
                    if self.verbose:
                        print(f"Scheduler: loading the request #{req.id}")

            # load memory
            if kv_size > 0:
                self.memory.mem_load(kv_size)
            
            total_len = 0
            init_cnt = 0
            for req in batch_req:
                if req.is_init:
                    total_len += req.input
                    init_cnt += 1
                    req.set_que_delay(current)
                else:
                    total_len += 1

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.get_batch_id(), batch_req[0].model, total_len, init_cnt, '1', current, kv_size, evict_size, load_size, True)
            # add alredy fired system
            batch.fired.append(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            if self.verbose:
                print(f"Scheduler: scheduling new batch #{batch.batch_id} to sys[{sys}]")
                print(f"Scheduler: batch #{batch.batch_id} has request #: ",end='')
                for req in batch.requests:
                    print(f"{req.id} ", end='')
                print()
            return batch
        
        # Schedule already batched request
        else:
            if len(self.inflight) == 0:
                return None
            else:
                batch = None
                # find batch
                for b in self.inflight:
                    if b.batch_id == batch_id:
                        batch = b
                if batch == None:
                    return None
                # check if this has been runned in the system
                if sys in batch.fired:
                    return None
                else:
                    batch.fired.append(sys)
                    if self.verbose:
                        print(f"Scheduler: scheduling exsisting batch #{batch.batch_id} to sys[{sys}]")
                    return batch

    # pop inflight, add to done
    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        req_cnt = 0
        if len(self.inflight) == 0:
            return 0, 0, 0
        batch = None
        # find batch
        id -= 1
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            return 0, 0, 0
        # already done
        if sys in batch.end:
            return 0, 0, 0
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            for i in range(self.npu_num):
                if i not in batch.end:
                    return 0, 0, 0
                
        if self.verbose:
            print(f"Scheduler: batch #{batch.batch_id} is done")
                
        pool = []
        for req in batch.requests:
            # change phase
            if req.is_init:
                req.is_init = False
                prompt_t += req.input
                gen_t += 1 # generated one token
                req.set_ttft(finish)

            else:
                gen_t += 1

            req.input += 1
            # check done
            if req.output <= req.input:
                if self.verbose:
                    print(f"Scheduler: request #{req.id} is done")
                # remove kv cache here
                kv_size = self.memory.get_evict_kv(req)
                self.memory.mem_store(kv_size)
                req.add_latency(finish)
                self.done.append(req)
                req_cnt += 1

            # return to pool
            else:
                pool.append(req)
        # return to request pool **at front**
        self.request = pool + self.request

        del self.inflight[idx]
        del batch
        return prompt_t, gen_t, req_cnt
    

    ##### Helper Functions ######
    # get new request id
    def get_req_id(self):
        self.req_ids += 1
        return self.req_ids

    # get new batch id
    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    # add a request
    def add_request(self, req, is_init=True):
        new = [self.get_req_id()]
        new_req = Request(*(new+req), is_init=is_init)
        self.request.append(new_req)
        return
    
    # get first request's arrival time
    def get_first_arrival_time(self):
        return self.request[0].arrival if self.request[0].arrival != 0 else 1 # need to add event handler at first

    # print results in done
    def print_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def is_request_empty(self):
        if len(self.request) == 0 and len(self.inflight) == 0:
            return True
        else:
            return False
        
    # save requests information to an output file
    def save_output(self, output_file):
        output_file = f'../{output_file}'
        with open(output_file, mode='w', newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            writer.writerow(['request id', 'model', 'input', 'output', 
                            'arrival', 'end_time', 'latency', 
                            'queuing_delay', 'TTFT', 'TPOT'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.id,
                    req.model,
                    req.input,
                    req.output,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot
                ])