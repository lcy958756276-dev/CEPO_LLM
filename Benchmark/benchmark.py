"""对LLM服务做极限压测的脚本
"""

import aiohttp
import asyncio
import json
import logging
import time
from typing import List, Tuple
import numpy as np


logger = logging.getLogger(__name__)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

API_URL = 'http://localhost:8000/chat'

HEADERS = {
    'Content-Type': 'application/json',
}


async def send_request(session, payload, prompt_len):#发送一次请求
    request_start_time = time.time()
    async with session.post(API_URL, data=payload, headers=HEADERS) as response:#发送HTTP请求发送HTTP请求，aiohttp + async
        if response.status == 200:
            result = await response.json()#解析返回
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            request_end_time = time.time()
            request_latency = request_end_time - request_start_time
            REQUEST_LATENCY.append((prompt_tokens, completion_tokens, request_latency))
            return result#(输入字符数, 输出字符数, 总耗时)
        else:
            return {'error': response.status, 'message': await response.text()}


class BenchMarkRunner:#压测调度器
    def __init__(
        self,
        requests: List[Tuple[str, int, int]],  # prompt, prompt_len, completion_len 所有请求
        concurrency: int,#并发数
    ):
        self.concurrency = concurrency
        self.requests = requests
        self.request_left = len(requests)
        self.request_queue = asyncio.Queue(concurrency or 100)#请求队列

    async def run(self):
        tasks = []
        for i in range(self.concurrency):
            tasks.append(asyncio.create_task(self.worker()))#创建concurrency个worker，每个worker都会不断处理请求
        for req in self.requests:
            await self.request_queue.put(req)#把请求放进队列
        await asyncio.gather(*tasks)#等待任务执行，直到worker完成

    # async def worker(self):#worker 是真正执行推理请求的地方,并发执行请求
    #     timeout = aiohttp.ClientTimeout(total=5 * 60)
    #     async with aiohttp.ClientSession(timeout=timeout) as session:#创建 HTTP session，一个worker会复用一个session，可以减少TCP链接开销
    #         while self.request_left > 0:
    #             prompt = await self.request_queue.get()#从队列取任务
    #             payload = json.dumps({
    #                                   "query":prompt,
    #                                   "history": [],
    #                                   })#构造请求

    #             response = await send_request(session, payload, len(prompt))
    #             self.request_left -= 1
    #             print(f"Response {len(self.requests) - self.request_left}")
    async def worker(self):
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    prompt = self.request_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                    
                payload = json.dumps({
                    "query": prompt,
                    "history": []
                })

                result = await send_request(session, payload, len(prompt))

                if "error" not in result:
                    print("done one request")

def main():
    concurrency = 25#50 # 并发数
    logger.info("Preparing for benchmark.")
    testset = json.load(open("./data/summary_test.json"))
    input_requests = [item["instruction"] for item in testset] #提取prompt

    logger.info("Benchmark starts.")
    benchmark_start_time = time.time()
    asyncio.run(BenchMarkRunner(input_requests, concurrency).run())
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time

    print(f"Total time: {benchmark_time:.4f} s")
    print(f"Throughtput(request/s): {len(REQUEST_LATENCY) / benchmark_time:.2f} requests/s")#Throughput = 请求数 / 时间   请求吞吐量   REQUEST_LATENCY单请求下的(输入字符数, 输出字符数, 总耗时)
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.4f} s")#平均单个请求端到端耗时
    avg_per_token_latency = np.mean(
        [
            latency / (prompt_len + output_len)
            for prompt_len, output_len, latency in REQUEST_LATENCY
        ]
    )
    print(f"Average latency per token: {avg_per_token_latency:.4f} s")#单个请求的平均 token 延迟 = 总耗时 / 总 token 数
    avg_per_output_token_latency = np.mean(
        [latency / max(1,output_len) for _, output_len, latency in REQUEST_LATENCY]
    )
    print("Average latency per output token: " f"{avg_per_output_token_latency:.4f} s")#单个请求的平均输出 token 延迟 = 总耗时 / 输出 token 数
    throughput = (
            sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time#单位时间内生成的“输出 token 数”
    )
    print(f"Throughput: {throughput} tokens/s")


if __name__ == '__main__':
    main()
