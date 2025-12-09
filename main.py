import os
import json
import hashlib
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import dashscope
import redis
import uvicorn

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# api配置
# 调用 dashscope.api_key = "sk-"
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# Redis连接
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
try:
    cache_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    cache_client.ping()
    print("Redis connected.")
except:
    print("Using local memory.")
    cache_client = None
    local_memory = {}


def get_cache(key):
    if cache_client: return cache_client.get(key)
    return local_memory.get(key)


def set_cache(key, val, ex=3600):
    if cache_client:
        cache_client.setex(key, ex, val)
    else:
        local_memory[key] = val


# === 业务逻辑 ===
def extract_text(file_bytes):
    import io
    text = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p in pdf.pages:
                t = p.extract_text()
                if t: text.append(t)
        return "\n".join(text)
    except Exception as e:
        print(f"PDF Error: {e}")
        return ""


def ask_ai(resume, jd):
    prompt = f"""
    你是一个面试官。请分析简历与JD的匹配度。
    简历：{resume[:2500]}
    JD：{jd}
    请 strictly 返回纯 JSON 格式 (不要 Markdown):
    {{
        "candidate_info": {{ "name": "姓名", "email": "邮箱", "education": "学历", "years_exp": "年限", "skills": ["技能1"] }},
        "match_analysis": {{ "score": 85, "summary": "评价", "missing_skills": ["缺失1"], "keyword_match": ["匹配1"] }}
    }}
    """
    try:
        resp = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'
        )
        if resp.status_code == 200:
            content = resp.output.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        return {"error": "AI API Error", "match_analysis": {"score": 0, "summary": resp.message}}
    except Exception as e:
        return {"error": str(e), "match_analysis": {"score": 0, "summary": "解析异常"}}


# === 接口 ===
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), jd: str = Form(...)):
    content = await file.read()
    key_str = f"{hashlib.md5(content).hexdigest()}_{hashlib.md5(jd.encode()).hexdigest()}"

    if cached := get_cache(key_str):
        print("Cache Hit")
        return json.loads(cached)

    text = extract_text(content)
    if not text:
        raise HTTPException(400, "PDF 内容为空或无法解析")

    result = ask_ai(text, jd)
    set_cache(key_str, json.dumps(result))
    return result


# 本地调试入口
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)