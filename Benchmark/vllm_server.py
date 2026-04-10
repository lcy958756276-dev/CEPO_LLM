import os 
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig,snapshot_download
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
import uuid
import json 
import copy 
def _build_prompt(tokenizer, query, history=None, system=""):
    messages = []

    if system:
        messages.append({"role": "system", "content": system})

    if history:
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": query})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    return None, prompt   # vLLM用token ids

# 停用词清理
def remove_stop_words(token_ids,stop_words_ids):
    token_ids=copy.deepcopy(token_ids)
    while len(token_ids)>0:
        if token_ids[-1] in stop_words_ids:
            token_ids.pop(-1)
        else:
            break
    return token_ids
# http接口服务
app=FastAPI()

# vLLM参数
model_dir = "qwen/Qwen1.5-0.5B-Chat"
tensor_parallel_size=1
gpu_memory_utilization=0.6
#quantization='gptq'#量化
quantization=None
dtype='float16'
generation_config = None
tokenizer = None
stop_words_ids = None
engine = None
# vLLM模型加载
def load_vllm():
    # global generation_config, tokenizer, stop_words_ids, engine       
    # 模型下载
    snapshot_download(model_dir)
    # 模型基础配置
    generation_config=GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
    # 加载分词器
    tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    # 处理 eos_token_id（兼容 list）
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]
    # tokenizer.eos_token_id=generation_config.eos_token_id
    # 推理终止词
    stop_words_ids=[eos_id]
    # vLLM基础配置
    args=AsyncEngineArgs(model_dir)
    args.worker_use_ray=False
    args.engine_use_ray=False
    args.tokenizer=model_dir
    args.tensor_parallel_size=tensor_parallel_size
    args.trust_remote_code=True
    args.quantization=quantization
    args.gpu_memory_utilization=gpu_memory_utilization
    args.dtype=dtype
    args.max_num_seqs=15    # batch最大20条样本
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE']='True'
    engine=AsyncLLMEngine.from_engine_args(args)
    return generation_config,tokenizer,stop_words_ids,engine


# 用户停止句匹配
def match_user_stop_words(response_token_ids,user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids)<len(stop_tokens):
            continue 
        if response_token_ids[-len(stop_tokens):]==stop_tokens:
            return True  # 命中停止句, 返回True
    return False

# chat对话接口
@app.post("/chat")
async def chat(request: Request):
    request=await request.json()
    
    query=request.get('query',None)
    history=request.get('history',[])
    system=request.get('system','You are a helpful assistant.')
    stream=request.get("stream",False)
    user_stop_words=request.get("user_stop_words",[])    # list[str]，用户自定义停止句，例如：['Observation: ', 'Action: ']定义了2个停止句，遇到任何一个都会停止
    
    if query is None:
        return Response(status_code=502,content='query is empty')

    # 用户停止词
    user_stop_tokens=[]
    for words in user_stop_words:
        user_stop_tokens.append(tokenizer.encode(words))
    
        # 构造prompt
    prompt_text, prompt_tokens = _build_prompt(
        tokenizer,
        query,
        history=history,
        system=system
    )
        
    # vLLM请求配置
    sampling_params=SamplingParams(stop_token_ids=stop_words_ids, 
                                    early_stopping=False,
                                    top_p=generation_config.top_p,
                                    top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                    temperature=generation_config.temperature,
                                    repetition_penalty=generation_config.repetition_penalty,
                                    max_tokens=generation_config.max_new_tokens)
    # vLLM异步推理（在独立线程中阻塞执行推理，主线程异步等待完成通知）
    request_id=str(uuid.uuid4().hex)
    results_iter=engine.generate(prompt=None,sampling_params=sampling_params,prompt_token_ids=prompt_tokens,request_id=request_id)
    
    # 流式返回，即迭代transformer的每一步推理结果并反复返回
    if stream:
        async def streaming_resp():
            async for result in results_iter:
                # 移除im_end,eos等系统停止词
                token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
                # 返回截止目前的tokens输出                
                text=tokenizer.decode(token_ids)
                yield (json.dumps({'text':text})+'\0').encode('utf-8')
                # 匹配用户停止词,终止推理
                if match_user_stop_words(token_ids,user_stop_tokens):
                    await engine.abort(request_id)   # 终止vllm后续推理
                    break
        return StreamingResponse(streaming_resp())

    # 整体一次性返回模式
    async for result in results_iter:
        # 移除im_end,eos等系统停止词
        token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
        # 返回截止目前的tokens输出                
        text=tokenizer.decode(token_ids)
        # 匹配用户停止词,终止推理
        if match_user_stop_words(token_ids,user_stop_tokens):
            await engine.abort(request_id)   # 终止vllm后续推理
            break

    ret={
        "text":text,
        "usage":
        {
            "prompt_tokens":len(prompt_tokens),
            "completion_tokens":len(token_ids)
        }
        }
    return JSONResponse(ret)





if __name__=='__main__':
    generation_config,tokenizer,stop_words_ids,engine=load_vllm()
    uvicorn.run(app,
                host="0.0.0.0",
                port=8000,
                log_level="debug")