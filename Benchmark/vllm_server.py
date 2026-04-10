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
# 按chatml格式构造千问的Prompt
def _build_prompt(
                generation_config,
                tokenizer,
                query,
                history=None,
                system=""):
    if history is None:
        history=[]

    # 包裹发言内容的token
    im_start,im_start_tokens='<|im_start|>',[tokenizer.im_start_id]
    im_end,im_end_tokens='<|im_end|>',[tokenizer.im_end_id]
    # 换行符token
    nl_tokens=tokenizer.encode("\n")

    # 用于编码system/user/assistant的一段发言, 格式{role}\n{content}
    def _tokenize_str(role,content): # 返回元组，下标0是文本，下标1是token ids
        return f"{role}\n{content}",tokenizer.encode(role)+nl_tokens+tokenizer.encode(content)
    
    # 剩余token数
    left_token_space=generation_config.max_window_size

    # prompt头部: system发言
    system_text_part,system_tokens_part=_tokenize_str("system", system) # system_tokens_part -->    system\nYou are a helpful assistant.
    system_text=f'{im_start}{system_text_part}{im_end}'
    system_tokens=im_start_tokens+system_tokens_part+im_end_tokens # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
    left_token_space-=len(system_tokens)
    
    # prompt尾部: user发言和assistant引导
    query_text_part,query_tokens_part=_tokenize_str('user', query)
    query_tokens_prefix=nl_tokens+ im_start_tokens
    query_tokens_suffix=im_end_tokens+nl_tokens+im_start_tokens+tokenizer.encode('assistant')+nl_tokens
    if len(query_tokens_prefix)+len(query_tokens_part)+len(query_tokens_suffix)>left_token_space: # query太长截断
        query_token_len=left_token_space-len(query_tokens_prefix)-len(query_tokens_suffix)
        query_tokens_part=query_tokens_part[:query_token_len]
        query_text_part=tokenizer.decode(query_tokens_part)
    query_tokens=query_tokens_prefix+query_tokens_part+query_tokens_suffix
    query_text=f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"
    left_token_space-=len(query_tokens)
    
    # prompt腰部: 历史user+assitant对话
    history_text,history_tokens='',[]
    for hist_query,hist_response in reversed(history):    # 优先采用最近的对话历史
        hist_query_text,hist_query_tokens_part=_tokenize_str("user",hist_query) # user\n历史提问
        hist_response_text,hist_response_tokens_part=_tokenize_str("assistant",hist_response) # assistant\n历史回答
        # 生成本轮对话
        cur_history_tokens=nl_tokens+im_start_tokens+hist_query_tokens_part+im_end_tokens+nl_tokens+im_start_tokens+hist_response_tokens_part+im_end_tokens
        cur_history_text=f"\n{im_start}{hist_query_text}{im_end}\n{im_start}{hist_response_text}{im_end}"
        # 储存多轮对话
        if len(cur_history_tokens)<=left_token_space:
            history_text=cur_history_text+history_text
            history_tokens=cur_history_tokens+history_tokens
            left_token_space-=len(cur_history_tokens)
        else:
            break 
            
    # 生成完整Prompt
    prompt_str=f'{system_text}{history_text}{query_text}'
    prompt_tokens=system_tokens+history_tokens+query_tokens
    return prompt_str,prompt_tokens

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

# vLLM模型加载
def load_vllm():
    global generation_config,tokenizer,stop_words_ids,engine    
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
    stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_id]
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

generation_config,tokenizer,stop_words_ids,engine=load_vllm()

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
    prompt_text,prompt_tokens=_build_prompt(generation_config,tokenizer,query,history=history,system=system)
        
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
    uvicorn.run(app,
                host="0.0.0.0",
                port=8000,
                log_level="debug")