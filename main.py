import os
import openai
import requests
import time
import json
import warnings
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from postgrest import SyncPostgrestClient
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from functools import lru_cache

warnings.filterwarnings('ignore')
load_dotenv()

# 1. 초기 설정 및 클라이언트
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

supabase = SyncPostgrestClient(
    f"{SUPABASE_URL}/rest/v1", 
    headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
)

app = FastAPI(title="Quant Hybrid RAG Server", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stock_text = """
당신은 승률 51%의 우위와 손익비(Risk/Reward) 1:3 이상을 노리는 냉혹한 실전 퀀트 트레이더입니다.
아래의 [실전 매매 원칙]과 [출력 포맷 강제 규정]을 철저히 지켜서 답변하세요.

[실전 퀀트 매매 원칙]
1. 비대칭 손익비: 손절은 짧게, 수익은 길게 열려있는 자리에서만 진입한다.
2. VCP & 다이버전스: 볼린저밴드 수축기, 거래량 바닥, RSI 상승 다이버전스는 세력의 개입과 추세 반전의 강력한 신호다.
3. 테마와 내러티브: 기술적 자리가 좋아도 시장의 주도 테마(돈이 몰리는 섹터)가 아니면 무의미하다.
4. 책임지는 타점: "현재 매수 타점 아님"으로 회피하지 마라. 하락 시 지지를 받을 수 있는 가격대(예: 20일/60일선, 전저점)를 찾아 "내가 트레이더라면 이 가격에 대기하겠다"는 [대기 매수 타점]을 반드시 제시하라.

[출력 포맷 강제 규정 🚨절대 엄수🚨]
- 모든 답변은 줄글 형태를 금지하며, 가독성을 극대화하기 위해 반드시 아래 예시와 100% 동일하게 `**` 마크다운을 사용하여 작성하세요.

--- (답변 출력 템플릿 예시) ---
### ■ 스토리 & 테마
- **테마 요약:** (현재 시장의 어떤 핵심 주도 테마와 엮여 있는지 설명)
- **시장 위치:** (이 테마가 초기인지, 확산기인지, 끝물인지 판단)
- **모멘텀:** (종목 상승의 거시적 촉매제)

### ■ 수급 & 거래량 추적
- **거래량 상태:** (평균 대비 거래량 수치 평가)
- **수급 해석:** (매집인지, 설거지인지 판단)

### ■ 차트 셋업 판독
- **현재 위치:** (이평선, 볼린저밴드 기준 위치)
- **보조 지표:** (RSI, 다이버전스 특이사항)
- **차트 패턴:** (특이 패턴 유무)

### ■ 기계적 타점 및 손익비
- **종합 판단:** (강력 매수 / 분할 매수 / 관망 후 대기매수 중 1)
- **진입 타점:** (숫자로 대기 매수 가격 명확히 제시)
- **목표가:** (저항선 기반 익절 가격)
- **손절가:** (지지선 이탈 시 자를 가격)
- **손익비 분석:** (손절폭 대비 수익폭 산출)
------------------------
"""

knowledge_text = """
[실전 트레이딩 셋업 및 지표 해석 사전]
1. 볼린저밴드 수축 (Squeeze) & 돌파
- 밴드폭이 10% 이하로 좁아지면 극단적인 변동성 수축기(VCP)다. 
- 수축기 이후 상단을 뚫으면 강력한 상승 랠리다.

2. RSI 다이버전스와 공포 매매
- 진짜 바닥은 주가는 하락해 전저점을 깼는데, RSI는 저점을 높이는 현상(상승 다이버전스)이다.

3. 거래량 클라이맥스 (Volume Climax)
- 바닥권 장대음봉에서 터진 대량 거래량은 세력의 패닉셀 매집이다.

4. 60일 이평선(수급선)의 생명력
- 60일선이 탄탄하게 우상향 중인데 20일선을 깨고 60일선까지 급락했다면 확률 높은 1차 매수 타점이다.
"""

class NewsData(BaseModel):
    title: str; content: str; source: str; url: str
class StockQuery(BaseModel):
    question: str
class IndexData(BaseModel):
    indexName: str; price: float

print("✅ KRX 종목 리스트 로딩 중...")
krx = fdr.StockListing('KRX')

def get_embedding(text: str):
    response = client.embeddings.create(input=text[:1000] if text else "", model="text-embedding-3-small")
    return response.data[0].embedding

@lru_cache(maxsize=128)
def find_stock(question: str):
    q_no_space = question.replace(" ", "")
    matched = krx[krx['Name'].apply(lambda n: n.replace(" ", "") in q_no_space)]
    if not matched.empty:
        best = matched.loc[matched['Name'].str.len().idxmax()]
        return best['Name'], best['Code']
    try:
        ticker_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "미국 주식 티커 심볼만 대문자로 반환. 없으면 NONE"}, {"role": "user", "content": question}],
            max_tokens=10
        )
        ticker = ticker_response.choices[0].message.content.strip()
        if ticker != "NONE" and len(ticker) <= 5:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName")
            if name: return name, f"US:{ticker}"
    except: pass
    return None, None

@lru_cache(maxsize=32)
def fetch_stock_data(code: str):
    if code.startswith("US:"):
        df = yf.download(code.replace("US:", ""), period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        return df, "달러"
    return fdr.DataReader(code).tail(200), "원"

def get_technical_analysis(name: str, code: str) -> str:
    try:
        df, unit = fetch_stock_data(code)
        if df.empty or len(df) < 60: return f"[{name} 데이터 부족]"
        
        close = df['Close']
        price = round(float(close.iloc[-1]), 2)
        ma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
        ma60 = round(float(close.rolling(60).mean().iloc[-1]), 2)
        
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = round(float((bb_mid + 2 * bb_std).iloc[-1]), 2)
        bb_lower = round(float((bb_mid - 2 * bb_std).iloc[-1]), 2)
        bb_width = round(float(((bb_upper - bb_lower) / float(bb_mid.iloc[-1])) * 100), 2)
        
        vol_ratio = round(float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]), 2)
        
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = float((100 - 100 / (1 + gain / loss)).iloc[-1])
        
        return (f"[{name} 기술적 지표]\n- 현재가: {price:,}{unit}\n- 20일선: {ma20:,} / 60일선: {ma60:,}\n"
                f"- RSI: {rsi:.1f}\n- 볼린저 밴드폭: {bb_width}%\n- 거래량: {vol_ratio}배 터짐\n")
    except Exception as e: return f"분석 실패: {e}"

@app.post("/collect")
def collect_news(data: NewsData):
    try:
        if len(supabase.table("news_data").select("id").eq("url", data.url).execute().data) > 0: return {"status": "skipped"}
        supabase.table("news_data").insert({**data.dict(), "embedding": get_embedding(f"{data.title}\n{data.content}")}).execute()
        return {"status": "success"}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/collect_index")
def collect_index(data: IndexData):
    try:
        # supabase.table("index_data").insert({"index_name": data.indexName, "price": data.price, "updated_at": datetime.now().isoformat()}).execute()
        return {"status": "success"}
    except Exception as e: return {"status": "error"}

@app.get("/fetch_naver_news")
def fetch_naver_news(query: str = "증시"):
    try:
        url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=10&sort=date"
        headers = {"X-Naver-Client-Id": os.getenv("NAVER_CLIENT_ID"), "X-Naver-Client-Secret": os.getenv("NAVER_CLIENT_SECRET")}
        for item in requests.get(url, headers=headers).json().get("items", []):
            collect_news(NewsData(title=item['title'].replace("<b>","").replace("</b>",""), content=item['description'].replace("<b>","").replace("</b>",""), source="Naver", url=item['link']))
            time.sleep(0.1)
    except: pass
    return {"status": "success"}

@app.get("/ask_ai")
async def ask_ai(question: str):
    try:
        query_vec = get_embedding(question)
        stock_name, stock_code = find_stock(question)
        technical_info = get_technical_analysis(stock_name, stock_code) if stock_code else ""

        history_res = supabase.rpc("match_history", {"query_embedding": query_vec, "match_threshold": 0.5, "match_count": 1}).execute()
        past_analysis = f"과거 질문: {history_res.data[0]['question']}\n과거 답변: {history_res.data[0]['answer']}" if history_res.data else ""

        search_res = supabase.rpc("match_news", {"query_embedding": query_vec, "match_threshold": 0.3, "match_count": 5}).execute()
        if len(search_res.data) < 2:
            fetch_naver_news(query=stock_name if stock_name else question)
            search_res = supabase.rpc("match_news", {"query_embedding": query_vec, "match_threshold": 0.3, "match_count": 5}).execute()
        context_news = "\n".join([f"- {n['title']}: {n['content']}" for n in search_res.data])

        async def event_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'stock_code': stock_code, 'stock_name': stock_name})}\n\n"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": stock_text},
                    {"role": "user", "content": f"[과거분석]\n{past_analysis}\n\n[뉴스]\n{context_news}\n\n[차트]\n{technical_info}\n\n질문: {question}"}
                ],
                stream=True, stream_options={"include_usage": True}
            )
            full_answer = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    word = chunk.choices[0].delta.content
                    full_answer += word
                    yield f"data: {json.dumps({'type': 'chunk', 'content': word})}\n\n"
            try: supabase.table("ai_analysis_history").insert({"question": question, "answer": full_answer, "embedding": get_embedding(question + " " + full_answer)}).execute()
            except: pass

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart_data")
def get_chart_data(code: str, name: str = ""):
    try:
        df, _ = fetch_stock_data(code)
        if df.empty or len(df) < 20: return {"status": "error", "message": "데이터 부족"}

        df = df.copy()
        df['ma5'] = df['Close'].rolling(5).mean()
        df['ma20'] = df['Close'].rolling(20).mean()
        df['ma60'] = df['Close'].rolling(60).mean()

        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['ma20'] + 2 * bb_std
        df['bb_lower'] = df['ma20'] - 2 * bb_std

        delta = df['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        df['rsi'] = 100 - 100 / (1 + gain / loss)

        # ⭐ 5일, 20일, 60일 다중 지지/저항선 추출
        support_5 = float(df['Low'].rolling(5).min().iloc[-1])
        resist_5 = float(df['High'].rolling(5).max().iloc[-1])
        support_20 = float(df['Low'].rolling(20).min().iloc[-1])
        resist_20 = float(df['High'].rolling(20).max().iloc[-1])
        support_60 = float(df['Low'].rolling(60).min().iloc[-1])
        resist_60 = float(df['High'].rolling(60).max().iloc[-1])

        df = df.dropna(subset=['ma20']).tail(400)

        res = []
        for idx, row in df.iterrows():
            price = float(row['Close'])
            res.append({
                "time": str(idx.date()) if hasattr(idx, 'date') else str(idx).split('T')[0],
                "open": float(row['Open']) if 'Open' in row else price, "high": float(row['High']) if 'High' in row else price,
                "low": float(row['Low']) if 'Low' in row else price, "close": price,
                "ma5": float(row['ma5']) if not pd.isna(row['ma5']) else None, "ma20": float(row['ma20']) if not pd.isna(row['ma20']) else None,
                "ma60": float(row['ma60']) if not pd.isna(row['ma60']) else None, "bb_upper": float(row['bb_upper']) if not pd.isna(row['bb_upper']) else None,
                "bb_lower": float(row['bb_lower']) if not pd.isna(row['bb_lower']) else None, "rsi": float(row['rsi']) if not pd.isna(row['rsi']) else None,
                "volume": float(row['Volume'])
            })

        latest = res[-1]
        vol_ma20 = df['Volume'].rolling(20).mean()
        recent_vol_ma = vol_ma20.iloc[-1]
        vol_ratio = float(df['Volume'].iloc[-1] / recent_vol_ma) if not pd.isna(recent_vol_ma) and recent_vol_ma > 0 else 1.0

        def safe_round(val): return round(val, 2) if val is not None and not pd.isna(val) else None

        return {
            "status": "success", "stock_code": code, "stock_name": name,
            "latest": {
                "price": latest['close'], "rsi": safe_round(latest['rsi']), "volume_ratio": safe_round(vol_ratio), "ma5": safe_round(latest['ma5']),
                "support_5": safe_round(support_5), "resist_5": safe_round(resist_5),
                "support_20": safe_round(support_20), "resist_20": safe_round(resist_20),
                "support_60": safe_round(support_60), "resist_60": safe_round(resist_60)
            },
            "history": res
        }
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/")
def serve_ui(): return FileResponse("index.html")