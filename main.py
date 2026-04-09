import os
import openai
import requests
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from postgrest import SyncPostgrestClient
from dotenv import load_dotenv
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# 1. 환경 변수 불러오기
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("환경 변수에 SUPABASE_URL과 SUPABASE_KEY가 설정되지 않았습니다.")

# 2. 가벼운 PostgREST 클라이언트로 DB 직접 연결
supabase = SyncPostgrestClient(
    f"{SUPABASE_URL}/rest/v1", 
    headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
)

# 3. FastAPI 앱 초기화
app = FastAPI(title="Stock AI Server", version="1.0")

class NewsData(BaseModel):
    title: str
    content: str
    source: str
    url: str

# OpenAI 클라이언트 초기화
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    safe_text = text[:1000] if text else ""
    
    response = client.embeddings.create(
        input=safe_text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

@app.get("/")
def read_root():
    return {"message": "AI 기반 증시 데이터 서버가 정상 작동 중입니다 🚀"}

@app.post("/collect")
def collect_news(data: NewsData):
    try:
        # [1단계] 중복 검사
        check_duplicate = supabase.table("news_data").select("id").eq("url", data.url).execute()
        
        if len(check_duplicate.data) > 0:
            return {"status": "skipped", "message": "이미 존재하는 데이터입니다.", "url": data.url}

        # [2단계] 필터링 (내용이 20자 미만이면 패스)
        if len(data.content) < 20:
            return {"status": "skipped", "message": "내용이 너무 짧아 필터링되었습니다."}

        # [3단계] AI 임베딩 생성 (제목 + 내용을 합쳐서 문맥을 풍부하게 만듭니다)
        text_to_embed = f"제목: {data.title}\n내용: {data.content}"
        vector = get_embedding(text_to_embed)

        # [4단계] DB 저장 (embedding 데이터 포함!)
        insert_data = {
            "title": data.title,
            "content": data.content,
            "source": data.source,
            "url": data.url,
            "embedding": vector  # 드디어 DB에 벡터 숫자가 들어갑니다!
        }
        
        response = supabase.table("news_data").insert(insert_data).execute()
        
        return {"status": "success", "message": "데이터 및 임베딩 저장 완료!", "data": response.data}

    except Exception as e:
        print(f"❌ 에러 발생: {str(e)}")
        return {"status": "error", "message": str(e)}

# 1. 지수 데이터를 위한 모델
class IndexData(BaseModel):
    name: str
    price: float

@app.post("/collect_index")
async def collect_index(data: IndexData):
    try:
        # DB에 저장할 데이터 준비
        insert_data = {
            "name": data.name,
            "price": data.price
        }

        # Supabase 테이블에 insert 실행
        response = supabase.table("market_indices").insert(insert_data).execute()
        
        print(f"✅ DB 저장 완료: {data.name} - {data.price}")
        return {"status": "success", "data": response.data}
    
    except Exception as e:
        print(f"❌ DB 저장 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 네이버 뉴스 수집 엔드포인트
@app.get("/fetch_naver_news")
def fetch_naver_news(query: str = "증시"):
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=10&sort=date"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    
    try:
        response = requests.get(url, headers=headers)
        news_items = response.json().get("items", [])
        
        results = []
        for item in news_items:
            clean_title = item['title'].replace("<b>", "").replace("</b>", "").replace("&quot;", "'")
            clean_desc = item['description'].replace("<b>", "").replace("</b>", "").replace("&quot;", "'")
            
            news_data = NewsData(
                title=clean_title,
                content=clean_desc,
                source="Naver",
                url=item['link']
            )
            
            res = collect_news(news_data)
            results.append(res)
            
            time.sleep(0.5) 
            
        return {"status": "success", "count": len(results), "details": results}
    
    except Exception as e:
        print(f"❌ 네이버 뉴스 수집 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analyze_trend")
def analyze_trend():
    # 1. 최근 1시간 내 저장된 뉴스 50개 가져오기
    # (Supabase 쿼리로 최근 데이터를 긁어옵니다)
    news_list = supabase.table("news_data").select("title").order("created_at", desc=True).limit(50).execute()
    titles = [n['title'] for n in news_list.data]
    
    # 2. AI에게 테마 분석 시키기
    prompt = f"다음 뉴스 제목들을 보고 현재 시장의 핵심 테마 3개와 호재/악재 여부를 분석해줘: {', '.join(titles)}"
    
    response = client.chat.completions.create(
        model="gpt-4o", # 또는 gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {"analysis": response.choices[0].message.content}   

@app.get("/market_summary")
def get_market_summary():
    # 1. DB에서 최근 뉴스 30개 가져오기
    news = supabase.table("news_data").select("title, content").order("created_at", desc=True).limit(30).execute()
    
    # 2. 뉴스들을 하나의 텍스트로 합치기
    all_news = "\n".join([f"제목: {n['title']}, 내용: {n['content']}" for n in news.data])
    
    # 3. AI에게 브리핑 요청
    prompt = f"""
    아래 뉴스들을 분석해서 현재 주식 시장의 주요 테마와 투자자들의 심리를 요약해줘.
    내용이 짧더라도 핵심 키워드를 중심으로 호재와 악재를 구분해줘.
    
    뉴스 데이터:
    {all_news}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o", # 또는 gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {"summary": response.choices[0].message.content}

@app.get("/ask_ai")
def ask_ai(question: str):
    # 1. 사용자의 질문을 숫자로 변환 (임베딩)
    query_embedding = get_embedding(question) # 기존에 만든 함수 재활용

    # 2. Supabase에서 질문과 가장 비슷한 뉴스 TOP 5 검색 (벡터 검색)
    # rpc 호출을 위해 Supabase에 match_news 함수가 미리 생성되어 있어야 합니다.
    rpc_params = {
        "query_embedding": query_embedding,
        "match_threshold": 0.3, # 유사도 50% 이상만
        "match_count": 5       # 상위 5개
    }
    
    search_result = supabase.rpc("match_news", rpc_params).execute()
    
    # 3. 검색된 뉴스들을 하나의 문맥(Context)으로 합치기
    context = "\n".join([f"[{n['source']}] {n['title']}: {n['content']}" for n in search_result.data])
    
    if not context:
        return {"answer": "죄송합니다. 관련 뉴스를 찾지 못했습니다."}

    # 4. LLM(GPT)에게 뉴스 내용을 바탕으로 대답 요청
    prompt = f"""
    당신은 전문 주식 분석가입니다. 아래 제공된 최신 뉴스 데이터를 바탕으로 사용자의 질문에 답변하세요.
    데이터에 없는 내용은 지어내지 마세요.
    
    [최신 뉴스 데이터]:
    {context}
    
    [사용자 질문]:
    {question}
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", # 저렴하고 빠른 모델
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": [n['url'] for n in search_result.data] # 출처 링크 제공
    }