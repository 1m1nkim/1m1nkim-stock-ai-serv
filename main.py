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
from collections import Counter

warnings.filterwarnings('ignore')
load_dotenv()

# ==========================================
# 1. 초기 설정 및 클라이언트 (Supabase & OpenAI)
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

supabase = SyncPostgrestClient(
    f"{SUPABASE_URL}/rest/v1", 
    headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
)

# ==========================================
# 2. 한국투자증권 (KIS) OpenAPI 설정
# ==========================================
KIS_APP_KEY = os.getenv("KIS_APP_KEY")
KIS_APP_SECRET = os.getenv("KIS_APP_SECRET")
KIS_URL = os.getenv("KIS_URL", "https://openapivts.koreainvestment.com:29443")

kis_access_token = None
kis_token_expired_at = None

def get_kis_token():
    global kis_access_token, kis_token_expired_at
    if kis_access_token and kis_token_expired_at and datetime.now() < kis_token_expired_at:
        return kis_access_token

    print("🔄 한국투자증권 API Access Token 발급 중...")
    url = f"{KIS_URL}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET
    }
    
    response = requests.post(url, headers=headers, json=body)
    res_data = response.json()
    
    if "access_token" in res_data:
        kis_access_token = res_data["access_token"]
        kis_token_expired_at = datetime.now() + timedelta(hours=23)
        print("✅ 한투증권 토큰 발급 완료!")
        return kis_access_token
    return None

# ==========================================
# 3. 글로벌 차트 데이터 캐시
# ==========================================
CHART_CACHE = {}

# ==========================================
# 4. FastAPI 앱 초기화 및 데이터 로딩
# ==========================================
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
"""

class NewsData(BaseModel): title: str; content: str; source: str; url: str
class IndexData(BaseModel): indexName: str; price: float

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

# ==========================================
# 5. 핵심 로직: YFinance 과거데이터 + KIS 최신데이터 병합엔진
# ==========================================
def fetch_merged_chart_data(code: str, interval: str):
    is_us = code.startswith("US:")
    ticker = code.replace("US:", "") if is_us else code
    is_intraday = interval.endswith('m')
    
    yf_int_map = {'1m':'1m', '3m':'1m', '5m':'5m', '15m':'15m', '60m':'60m', '1d':'1d', '1w':'1d', '1M':'1d'}
    yf_per_map = {'1m':'7d', '3m':'7d', '5m':'60d', '15m':'60d', '60m':'730d', '1d':'2y', '1w':'5y', '1M':'10y'}
    
    yf_int = yf_int_map.get(interval, '1d')
    yf_per = yf_per_map.get(interval, '2y')
    
    yf_ticker = ticker
    if not is_us:
        try:
            market = krx.loc[krx['Code'] == ticker, 'Market'].values[0]
            yf_ticker = f"{ticker}.KS" if 'KOSPI' in str(market) else f"{ticker}.KQ"
        except:
            yf_ticker = f"{ticker}.KS"
            
    df_yf = pd.DataFrame()
    try:
        # 💡 미장 프리/애프터장 포함 옵션 유지
        df_yf = yf.download(yf_ticker, period=yf_per, interval=yf_int, prepost=is_us, progress=False)
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex): 
                df_yf.columns = df_yf.columns.droplevel(1)
            if getattr(df_yf.index, 'tz', None) is not None:
                df_yf.index = df_yf.index.tz_convert('Asia/Seoul').tz_localize(None)
    except Exception as e:
        print("YFinance Fetch Error:", e)

    df_kis = pd.DataFrame()
    token = get_kis_token()
    if token and not is_us:
        try:
            if not is_intraday:
                url = f"{KIS_URL}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
                headers = { "content-type": "application/json", "authorization": f"Bearer {token}", "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET, "tr_id": "FHKST03010100", "custtype": "P" }
                params = { "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker, "FID_INPUT_DATE_1": "", "FID_INPUT_DATE_2": "", "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1" }
                res = requests.get(url, headers=headers, params=params).json()
                if res.get('rt_cd') == '0' and res.get('output2'):
                    df_kis = pd.DataFrame(res['output2'])
                    df_kis = df_kis[['stck_bsop_date', 'stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_clpr', 'acml_vol']]
                    df_kis.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df_kis['Datetime'] = pd.to_datetime(df_kis['Date'], format='%Y%m%d')
                    df_kis.set_index('Datetime', inplace=True)
                    df_kis.drop(['Date'], axis=1, inplace=True)
            else:
                url = f"{KIS_URL}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
                headers = { "content-type": "application/json", "authorization": f"Bearer {token}", "appkey": KIS_APP_KEY, "appsecret": KIS_APP_SECRET, "tr_id": "FHKST03010200", "custtype": "P" }
                params = { "FID_ETC_CLS_CODE": "", "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker, "FID_INPUT_HOUR_1": "153000", "FID_PW_DATA_INCU_YN": "N" }
                res = requests.get(url, headers=headers, params=params).json()
                if res.get('rt_cd') == '0' and res.get('output2'):
                    df_kis = pd.DataFrame(res['output2'])
                    df_kis = df_kis[['stck_bsop_date', 'stck_cntg_hour', 'stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_prpr', 'cntg_vol']]
                    df_kis.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df_kis['Datetime'] = pd.to_datetime(df_kis['Date'] + df_kis['Time'], format='%Y%m%d%H%M%S')
                    df_kis.set_index('Datetime', inplace=True)
                    df_kis.drop(['Date', 'Time'], axis=1, inplace=True)
                    
            if not df_kis.empty:
                df_kis = df_kis.astype(float).sort_index()
        except Exception as e:
            print("KIS Patch Error:", e)

    if not df_yf.empty and not df_kis.empty:
        df = pd.concat([df_yf, df_kis])
        df = df[~df.index.duplicated(keep='last')].sort_index()
    elif not df_yf.empty: df = df_yf
    elif not df_kis.empty: df = df_kis
    else: df = pd.DataFrame()

    if df.empty and not is_intraday:
        df = fdr.DataReader(ticker).tail(500)

    if not df.empty:
        if is_intraday:
            rule = interval.replace('m', 'min')
            df = df.resample(rule, label='left', closed='left').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        else:
            if interval == '1w': df = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            elif interval == '1M': df = df.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()

    return df, "달러" if is_us else "원"

def get_data_with_ttl_cache(code: str, interval: str):
    global CHART_CACHE
    now = time.time()
    cache_key = f"{code}_{interval}"
    
    if cache_key in CHART_CACHE:
        cached_df, timestamp = CHART_CACHE[cache_key]
        if now - timestamp < 60:
            return cached_df.copy(), "달러" if code.startswith("US:") else "원"
            
    df, unit = fetch_merged_chart_data(code, interval)
    if not df.empty: CHART_CACHE[cache_key] = (df, now)
    return df, unit

def get_technical_analysis(name: str, code: str) -> str:
    try:
        df, unit = get_data_with_ttl_cache(code, "1d") 
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
        
        return (f"[{name} 일봉 기준 기술적 지표]\n- 현재가: {price:,}{unit}\n- 20선: {ma20:,} / 60선: {ma60:,}\n"
                f"- RSI: {rsi:.1f}\n- 볼린저 밴드폭: {bb_width}%\n- 거래량: {vol_ratio}배 터짐\n")
    except Exception as e: return f"분석 실패: {e}"

# ==========================================
# 6. FastAPI 엔드포인트
# ==========================================
@app.post("/collect")
def collect_news(data: NewsData):
    try:
        if len(supabase.table("news_data").select("id").eq("url", data.url).execute().data) > 0: return {"status": "skipped"}
        supabase.table("news_data").insert({**data.dict(), "embedding": get_embedding(f"{data.title}\n{data.content}")}).execute()
        return {"status": "success"}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/collect_index")
def collect_index(data: IndexData): return {"status": "success"}

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
def get_chart_data(code: str, name: str = "", interval: str = "1d"):
    try:
        # 1. 메인 차트 데이터 (현재 보고 있는 타임프레임)
        df, _ = get_data_with_ttl_cache(code, interval)
        
        # 2. 💡 피봇 & 고정 지지저항을 위해 '일봉(1d)' 데이터를 무조건 백그라운드에서 가져옵니다.
        daily_df, _ = get_data_with_ttl_cache(code, "1d")

        if df.empty or len(df) < 2: return {"status": "error", "message": "데이터 부족"}

        df = df.copy()
        
        # --- 유동 지표 (보고 있는 차트에 맞춰 변함) ---
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

        # --- 💡 고정 지표 (무조건 일봉 기준으로 계산되어 모든 분봉 차트에서 동일한 위치에 고정됨) ---
        if not daily_df.empty and len(daily_df) >= 2:
            # 기간 지지/저항 (일봉 기준)
            support_5 = float(daily_df['Low'].rolling(5).min().iloc[-1])
            resist_5 = float(daily_df['High'].rolling(5).max().iloc[-1])
            support_20 = float(daily_df['Low'].rolling(20).min().iloc[-1])
            resist_20 = float(daily_df['High'].rolling(20).max().iloc[-1])
            support_60 = float(daily_df['Low'].rolling(60).min().iloc[-1])
            resist_60 = float(daily_df['High'].rolling(60).max().iloc[-1])

            # 당일 피봇 및 지지/저항 (어제(전일) 일봉 캔들 기준)
            prev_day = daily_df.iloc[-2]
            prev_high = float(prev_day['High'])
            prev_low = float(prev_day['Low'])
            prev_close = float(prev_day['Close'])
            
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
        else:
            support_5 = resist_5 = support_20 = resist_20 = support_60 = resist_60 = None
            pivot = r1 = s1 = r2 = s2 = None

        df = df.dropna(subset=['ma20'])

        res = []
        is_intraday = interval.endswith('m')
        for idx, row in df.iterrows():
            price = float(row['Close'])
            res.append({
                "time": idx.isoformat() if is_intraday else idx.strftime('%Y-%m-%d'), 
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
                # 고정 지표들을 프론트로 쏴줍니다
                "support_5": safe_round(support_5), "resist_5": safe_round(resist_5),
                "support_20": safe_round(support_20), "resist_20": safe_round(resist_20),
                "support_60": safe_round(support_60), "resist_60": safe_round(resist_60),
                "pivot": safe_round(pivot), "s1": safe_round(s1), "s2": safe_round(s2), "r1": safe_round(r1), "r2": safe_round(r2)
            },
            "history": res
        }
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/fetch_dart")
def fetch_dart():
    try:
        dart_api_key = os.getenv("DART_API_KEY")
        if not dart_api_key:
            return {"status": "error", "message": "DART_API_KEY가 설정되지 않았습니다."}

        # 오늘 날짜 기준으로 검색
        today = datetime.now().strftime("%Y%m%d")
        url = "https://opendart.fss.or.kr/api/list.json"
        
        # pblntf_ty=B (주요사항보고서: 유/무상증자, 주식취득 등), I (수시공시: 수주, 영업잠정실적 등)
        for doc_type in ['B', 'I']: 
            params = {
                "crtfc_key": dart_api_key,
                "bgn_de": today,
                "end_de": today,
                "pblntf_ty": doc_type, 
                "page_count": 100
            }
            res = requests.get(url, params=params).json()
            
            if res.get('status') == '000' and 'list' in res:
                for item in res['list']:
                    # 호재성 키워드가 포함된 공시만 필터링 (순도 100% 유지)
                    title = item['report_nm']
                    good_keywords = ["공급계약", "주식취득", "무상증자", "영업잠정실적", "타법인주식", "공개매수"]
                    
                    if any(k in title for k in good_keywords):
                        corp_name = item['corp_nm']
                        content = f"[{corp_name}] 전자공시: {title}"
                        link = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcp_no']}"
                        
                        # DB 중복 체크 후 저장 (기존 collect_news 로직 활용)
                        if len(supabase.table("news_data").select("id").eq("url", link).execute().data) == 0:
                            supabase.table("news_data").insert({
                                "title": f"[공시] {corp_name} - {title}",
                                "content": content,
                                "source": "DART",
                                "url": link,
                                "embedding": get_embedding(f"{corp_name} {title}")
                            }).execute()

        return {"status": "success", "message": "DART 공시 수집 완료"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/fetch_kis_condition")
def fetch_kis_condition(seq: str = "1"): 
    # seq는 HTS에서 저장한 조건검색식의 번호입니다 (예: 0, 1, 2...)
    try:
        token = get_kis_token()
        if not token:
            return {"status": "error", "message": "한투 토큰 발급 실패"}

        url = f"{KIS_URL}/uapi/domestic-stock/v1/quotations/psearch-result"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": KIS_APP_KEY,
            "appsecret": KIS_APP_SECRET,
            "tr_id": "HHKST03900400", # 실시간 조건검색 TR ID
            "custtype": "P"
        }
        params = {
            "user_id": os.getenv("KIS_USER_ID"), 
            "seq": seq, 
            "mac_address": os.getenv("MAC_ADDRESS").replace(":", "").replace("-", "").lower(), # 특수문자 제거
            "bpass_chk_yn": "N",
            "clear_cmd_yn": "Y",
            "pblc_cmd_yn": "N",
            "out_func": "F",
            "ent_expt_cmd_yn": "N",
            "expt_cmd_yn": "N"
        }

        res = requests.get(url, headers=headers, params=params).json()
        
        # 💡 rt_cd가 '0'이면 무조건 통신 성공!
        if res.get('rt_cd') == '0':
            output2 = res.get('output2', [])
            
            # 포착된 종목이 없을 경우
            if not output2:
                return {"status": "success", "message": "현재 조건식에 포착된 종목이 없습니다.", "data": []}
            
            # 포착된 종목이 있을 경우
            detected_stocks = []
            for item in output2:
                stock_name = item['name']
                stock_code = item['code']
                detected_stocks.append(f"{stock_name}({stock_code})")
            
            # 조건검색에 포착된 종목들을 DB에 리포트로 저장
            title = f"[조건검색 포착] {len(detected_stocks)}종목 발굴"
            content = "포착 종목: " + ", ".join(detected_stocks)
            
            # 💡 위에서 추가했던 related_stocks 태그 컬럼에도 종목을 넣어줍니다.
            supabase.table("news_data").insert({
                "title": title,
                "content": content,
                "source": "KIS_CONDITION",
                "url": f"kis_cond_{datetime.now().strftime('%Y%m%d%H%M')}",
                "related_stocks": ", ".join(detected_stocks), # 태그 저장
                "embedding": get_embedding(content)
            }).execute()
                
            return {"status": "success", "message": f"{len(detected_stocks)}개 종목 포착", "data": detected_stocks}
            
        return {"status": "error", "message": res.get('msg1')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/ranked_stocks")
async def get_ranked_stocks():
    try:
        # 1. 최근 DB 데이터 50개 긁어오기 (공시, 찌라시, 조건검색 모두 포함)
        res = supabase.table("news_data").select("title, content, source, related_stocks").order("id", desc=True).limit(50).execute()
        if not res.data:
            return {"status": "error", "message": "DB에 데이터가 없습니다."}

        # 2. 💡 방금 만든 '태그(related_stocks)'를 활용해 가장 많이 겹치는 핫한 종목 5개 추출!
        all_stocks = []
        for row in res.data:
            if row.get('related_stocks') and row['related_stocks'].strip() != 'NONE':
                # 콤마로 분리해서 리스트에 추가
                stocks = [s.strip() for s in row['related_stocks'].split(',')]
                all_stocks.extend(stocks)
        
        if not all_stocks:
             return {"status": "error", "message": "데이터에서 종목 태그를 찾지 못했습니다."}

        # 가장 많이 언급된 종목 Top 5 뽑기
        top_5_tuples = Counter(all_stocks).most_common(5)
        target_stocks = [t[0] for t in top_5_tuples] # 예: ['삼성전자(005930)', '하이트진로(000080)', ...]

        # 3. 추출된 Top 5 종목의 최신 기술적 지표(차트) 수집
        stock_contexts = []
        recent_news_text = ""
        
        for stock_str in target_stocks:
            # "종목명(종목코드)" 형태에서 파싱
            if "(" in stock_str and ")" in stock_str:
                name = stock_str.split("(")[0]
                code = stock_str.split("(")[1].replace(")", "")
            else:
                name, code = find_stock(stock_str)
            
            if code:
                tech_info = get_technical_analysis(name, code)
                stock_contexts.append(f"[{name}({code})]\n{tech_info}")
                
                # 이 종목이 포함된 최근 뉴스/공시 텍스트도 컨텍스트에 추가
                relevant_news = [r for r in res.data if name in (r.get('related_stocks') or '')][:2]
                for rn in relevant_news:
                    recent_news_text += f"- [{rn['source']}] {rn['title']}\n"

        if not stock_contexts:
             return {"status": "error", "message": "종목 기술적 지표를 불러오지 못했습니다."}

        combined_tech = "\n\n".join(stock_contexts)

        # 4. AI 퀀트 스코어링 (무조건 JSON 뱉기)
        ranking_prompt = f"""
        당신은 AI 퀀트 엔진입니다. 아래의 [최근 호재/데이터]와 [차트 지표]를 분석하여 각 종목의 단기 상승 확률을 0~100점으로 평가하세요.
        반드시 아래 JSON 형식으로만 응답하세요.

        [최근 호재/데이터]
        {recent_news_text}

        [차트 지표]
        {combined_tech}

        [출력 포맷 (반드시 JSON)]
        {{
          "stocks": [
            {{
              "code": "005930",
              "name": "삼성전자",
              "score": 85,
              "reason": "조건검색 포착 및 20일선 지지 확인",
              "target_price": 85000,
              "stop_loss": 78000
            }}
          ]
        }}
        """

        score_res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": ranking_prompt}],
            response_format={ "type": "json_object" } # JSON 강제!
        )

        raw_json = score_res.choices[0].message.content
        parsed_data = json.loads(raw_json)
        stocks_list = parsed_data.get("stocks", [])
        sorted_stocks = sorted(stocks_list, key=lambda x: x.get('score', 0), reverse=True)

        return {"status": "success", "data": sorted_stocks}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/debug_mac")
def debug_mac():
    raw = os.getenv("MAC_ADDRESS", "없음")
    cleaned = raw.replace(":", "").replace("-", "")
    return {
        "raw": raw,
        "cleaned": cleaned,
        "length": len(cleaned)  # 반드시 12자리여야 함
    }

@app.get("/")
def serve_ui(): return FileResponse("index.html")