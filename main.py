import os
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from postgrest import SyncPostgrestClient
from dotenv import load_dotenv

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

        # [3단계] DB 저장
        insert_data = {
            "title": data.title,
            "content": data.content,
            "source": data.source,
            "url": data.url
        }
        
        response = supabase.table("news_data").insert(insert_data).execute()
        
        return {"status": "success", "message": "데이터가 성공적으로 저장되었습니다.", "data": response.data}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

# OpenAI 클라이언트 초기화
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str):
    safe_text = text[:1000] if text else ""
    
    response = client.embeddings.create(
        input=safe_text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

@app.post("/analyze_news")
async def analyze_news(news_id: int):
    # 1. DB에서 뉴스 가져오기
    news = supabase.table("news_data").select("*").eq("id", news_id).single().execute()
    content = news.data['content']
    
    # 2. 임베딩 생성 (AI가 이해하는 숫자로 변환)
    vector = get_embedding(content)
    
    # 3. DB에 업데이트 (embedding 컬럼에 저장)
    supabase.table("news_data").update({"embedding": vector}).eq("id", news_id).execute()
    
    return {"status": "success", "message": "임베딩 완료"}        