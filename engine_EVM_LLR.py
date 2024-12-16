import tensorrt as trt
from logzero import logger
from pathlib import Path
import threading
from queue import Queue
from dataloader import SampleDataLoader
from processor import Processor
import time


class Worker(threading.Thread):
    def __init__(self, ch_engine, pn_engine, input_queue, output_queue, device_id):
        super(Worker, self).__init__()
        self.processor = Processor(ch_engine, pn_engine, device_id=device_id, ret_eq_signal=False, ret_evm=False)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True

    def run(self):
        while self.running:
            data = self.input_queue.get()
            if data is None:
                self.running = False
                break
            result = self.processor.run(data)
            self.output_queue.put(result)


class Engine(object):
    def __init__(self, num_worker):
        self.ch_engine = self.load_engine('transformer_ch')
        self.pn_engine = self.load_engine('transformer_pn')
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.worker = []
        for _ in range(num_worker):
            worker = Worker(self.ch_engine, self.pn_engine, self.input_queue, self.output_queue, device_id=0)
            self.worker.append(worker)

    def load_engine(self, file_name):
        with trt.Logger(trt.Logger.ERROR) as logger, trt.Runtime(logger) as runtime:
            engine_path = Path(__file__).parents[0].resolve() / 'tensorrt_model' / (file_name + '.engine')
            with open(engine_path, mode='rb') as f:
                engine_bytes = f.read()
            engine = runtime.deserialize_cuda_engine(engine_bytes)
        return engine

    def worker_test(self, data_loader):
        for worker in self.worker:
            worker.start()
        start = time.time()
        for data in data_loader:
            self.input_queue.put(data)
        num_task = len(data_loader)
        received_result = 0
        while received_result < num_task:
            result = self.output_queue.get()
            received_result += 1
        end = time.time()
        for _ in self.worker:
            self.input_queue.put(None)
        logger.info(f'Number of completed tasks: {received_result}, Elapsed time: {end - start:.5f} sec, '
                    f'Throughput: {received_result / (end - start):.5f} Hz')


if __name__ == "__main__":
    num_worker = 2
    en = Engine(num_worker=num_worker)
    data_loader = SampleDataLoader('sample_no_noise', num_repeat=10)
    en.worker_test(data_loader=data_loader)