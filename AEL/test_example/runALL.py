import multiprocessing
import subprocess

# 定义要运行的 Python 命令
commands = [
    # "python joaov2.py --DS PROTEINS",
    # "python joaov2.py --DS DD",
    # "python joaov2.py --DS COLLAB",
    # "python joaov2.py --DS MUTAG",
    "python joaov2.py --DS REDDIT-BINARY",
    "python joaov2.py --DS REDDIT-MULTI-5K",
    "python joaov2.py --DS NCI1",
    "python joaov2.py --DS IMDB-BINARY",
]

def run_command(command):
    """运行单个命令"""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Command '{command}' finished with output:\n{result.stdout.decode()}")
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error:\n{e.stderr.decode()}")

if __name__ == "__main__":
    # 创建一个进程池，限制最大进程数为3
    with multiprocessing.Pool(processes=3) as pool:
        # 使用进程池并行运行命令
        pool.map(run_command, commands)
