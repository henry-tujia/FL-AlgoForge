from torch.multiprocessing import Queue,set_start_method
import data_preprocessing.custom_multiprocess as cm

class Client():
    def __init__(self,name) -> None:
        self.name = name
    def _print(self):
        for item in self.name:
            print(item, flush=True)

class Client2():
    def __init__(self,name) -> None:
        self.name = name
    def _print(self):
        for item in self.name:
            print(item, flush=True)

class Client_all():
    def __init__(self,client,name) -> None:
        if client =="0":
            self.client = Client(name)
        else:
            self.client = Client2(name)

def init_p(info,cli):
    global client
    info_inner = info.get()
    client = cli(info_inner[0],info_inner)

def run_clis(i):
    client.client._print()
    # print(client.name)
    # pass

if __name__ == "__main__":
    set_start_method('spawn')
    client_info = Queue()

    for i in range(2):  
        client_info.put([i]+[x*(i+1) for x in range(1,11)])
    # print(client_info.get())
    pool = cm.DreamPool(2, init_p,(client_info, Client_all))
    # pool2 = cm.DreamPool(2, init_p,(client_info, Client2))
    #用两个进程,初始化了一共20个实例
    
    pool.map(run_clis,range(10))
    #跑10个进程，一共多少个实例呢？

    pool.close()
    pool.join()