import multiprocessing

class MyClient:
    def __init__(self, client_id):
        self.client_id = client_id
    
    def run(self):
        print("Client ID:", self.client_id)

# 初始化函数，在每个进程中执行
def initialize_client(client_id):
    global client
    client = MyClient(client_id)  # 创建client实例

# 定义进程中的任务
def run_task():
    client.run()  # 使用全局client实例执行任务

# 创建进程池
process_pool = multiprocessing.Pool()

# 提交任务到进程池
for i in range(4):  # 提交4个任务
    client_id = "Client-" + str(i)  # 创建唯一的client id
    process_pool.apply_async(initialize_client, args=(client_id,))  # 初始化client实例
    process_pool.apply_async(run_task)  # 执行任务

# 关闭进程池，等待任务完成
process_pool.close()
process_pool.join()