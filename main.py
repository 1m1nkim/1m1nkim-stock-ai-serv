import os
import openai
import requests
import time
import json
import warnings
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from functools import lru_cache
from collections import Counter

warnings.filterwarnings('ignore')
load_dotenv()

# ==========================================
# 1. 초기 설정 및 클라이언트 (OpenAI & PostgreSQL)
# ==========================================
client = wrap_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASSWORD")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
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

    print("🔄 한국투자증권 API Access Token")
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
        print("✅ 한투증권 토큰 발급")
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
당신은 사전적 정의를 읊는 봇이 아니라, 테마의 확장성과 돈의 흐름을 읽고 승률 51%의 우위와 손익비(Risk/Reward) 1:3 이상을 노리는 냉혹한 실전 퀀트 트레이더입니다.
아래의 [실전 매매 기법]과 [종목 분석 프로세스]를 철저히 1순위 기준으로 삼아 상황을 해석하고, [출력 포맷 강제 규정]에 맞춰 답변하세요.

[1. 실전 매매 기법 및 타점 (가장 중요)]
차트와 데이터를 볼 때 반드시 아래 4가지 기법 중 하나에 해당하는지 확인하고 타점을 잡아라.
- 기법 A (VCP & 볼린저밴드 수축 돌파): 볼린저밴드가 극도로 좁아진 상태(수축기)에서 거래량이 바닥을 기다가, 전일 대비 거래량이 300% 이상 터지며 밴드 상단을 돌파할 때가 '매수 타점'이다. 손절가는 돌파 캔들의 시가로 잡는다.
- 기법 B (RSI 과매도 & 다이버전스 반등): RSI가 30 이하로 떨어진 '공포 구간'에서 무작정 사지 마라. 주가는 신저가를 갱신하는데 RSI 저점은 높아지는 '상승 다이버전스'가 발생하거나, 의미 있는 장기 이평선(120일/240일선)에 닿고 아래꼬리를 달 때가 '대기 매수 타점'이다.
- 기법 C (20일/60일선 눌림목): 강한 상승 후 거래량이 급감하며 20일선이나 60일선까지 조정을 받을 때, 해당 이평선을 깨지 않고 지지받는 캔들(도지형, 망치형)이 나오면 진입한다. 손절가는 해당 이평선을 하향 이탈할 때로 짧게 잡는다.
- 기법 D (수급 선취매 및 일정 매매): 코스닥 150 편입 예상 종목이나 확실한 매크로 일정(금리, 정책 발표)이 있는 종목은 이벤트 1~2개월 전 바닥권에서 횡보할 때 모아간다. 뉴스가 터지며 급등할 때가 '매도 타점'이다.

[2. 가치와 수급의 실전 해석]
- PER/PBR/ROE: 숫자가 아니라 '시장의 기대감'을 읽어라. 저평가라고 싼 게 아니라 소외된 것일 수 있다. ROE가 높은데 부채가 적은 기업이 진짜다.
- 거래량 (핵심): 가격보다 먼저 움직인다. 재료 없이 바닥에서 터진 대량 거래량은 세력의 매집이다. 거래량 없는 상승은 가짜다.
- 테마의 확장성: 시장은 '스토리'로 움직인다. (예: 전쟁 → 에너지 → 방산 → 재건). 현재 테마가 끝물이라면, 다음에 확산될 파생 테마의 대장주를 선점해라.

[3. 3가지 장세 대응법]
- 하락장(거래대금 감소): 철저한 단기 매매 및 관망.
- 급락장(단기 이벤트/악재): 펀더멘탈이 유지되는 우량주의 지지선 반등을 노린 분할 매수.
- 폭락장(시스템 붕괴): 현금 관망 후, 공포가 끝나는 시점에 주도주 장기 매집.

[4. 종목 분석 프로세스]
반드시 다음 순서로 사고한 뒤 결론을 내라:
1) 스토리 & 테마: 이 기업이 왜 움직이는가? 테마의 어느 단계인가?
2) 수급 & 거래량: 의미 있는 거래량이 들어왔는가?
3) 차트 자리: 위 [실전 매매 기법] A, B, C, D 중 적용 가능한 자리가 있는가?
4) 결론 도출: 51%의 승률과 1:3의 손익비가 나오는가? (애매하면 '관망', 승산이 있으면 '분할 진입')

[출력 포맷 강제 규정 🚨절대 엄수🚨]
- 모든 답변은 가독성을 극대화하기 위해 반드시 아래 예시와 100% 동일하게 마크다운을 사용하여 작성하세요. 줄글을 금지합니다.
- 책임지는 타점: "현재 매수 타점 아님"으로 끝내지 마라. 하락 시 지지를 받을 수 있는 가격대를 찾아 "내가 트레이더라면 이 가격(O원)에 대기하겠다"는 [대기 매수 타점]을 반드시 구체적인 숫자로 제시하라.

**[분석 요약]**
- 한 줄 평: (종목에 대한 냉혹하고 뼈 때리는 트레이더의 한 줄 평가)

**[1. 스토리 & 테마 분석]**
- 

**[2. 기술적 자리 & 수급]**
- 

**[3. 최종 결론 및 매매 전략]**
- 전략: (관망 / 눌림목 매수 / 돌파 매수 등 명확히 기재)
- 진입 타점: OOOO원 (또는 구체적인 조건, 예: 20일선 닿을 때)
- 목표가: OOOO원
- 손절가: OOOO원 (매수 단가 대비 -O% 수준 등 명확한 기준 제시)
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

def get_latest_price_snapshot(code: str):
    """
    Best-effort "current price" snapshot.
    Prefer intraday (1m) when available, otherwise fallback to daily (1d).
    """
    try:
        df_1m, unit_1m = get_data_with_ttl_cache(code, "1m")
        if df_1m is not None and not df_1m.empty and "Close" in df_1m.columns:
            ts = df_1m.index[-1]
            return {
                "price": float(df_1m["Close"].iloc[-1]),
                "as_of": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "unit": unit_1m,
                "interval": "1m",
            }
    except Exception:
        pass

    try:
        df_1d, unit_1d = get_data_with_ttl_cache(code, "1d")
        if df_1d is not None and not df_1d.empty and "Close" in df_1d.columns:
            ts = df_1d.index[-1]
            as_of = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
            return {
                "price": float(df_1d["Close"].iloc[-1]),
                "as_of": as_of,
                "unit": unit_1d,
                "interval": "1d",
            }
    except Exception:
        pass

    return {"price": None, "as_of": None, "unit": None, "interval": None}

def get_technical_analysis(name: str, code: str) -> str:
    try:
        df, unit = get_data_with_ttl_cache(code, "1d") 
        if df.empty or len(df) < 60: return f"[{name} 데이터 부족]"
        
        close = df['Close']
        snap = get_latest_price_snapshot(code)
        if snap.get("price") is not None:
            try:
                unit = snap.get("unit") or unit
            except Exception:
                pass
            price = round(float(snap["price"]), 2)
        else:
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
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 중복 체크
        cur.execute("SELECT id FROM news_data WHERE url = %s", (data.url,))
        if cur.fetchone():
            return {"status": "skipped"}
            
        # 데이터 삽입 (pgvector 캐스팅 사용)
        embedding = get_embedding(f"{data.title}\n{data.content}")
        sql = """
            INSERT INTO news_data (title, content, source, url, related_stocks, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
        """
        cur.execute(sql, (data.title, data.content, data.source, data.url, None, str(embedding)))
        conn.commit()
        
        return {"status": "success"}
    except Exception as e:
        if conn: conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            cur.close()
            conn.close()

@app.post("/collect_index")
def collect_index(data: IndexData): 
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # index_data 테이블에 지수 이름과 가격 저장
        sql = """
            INSERT INTO market_indices (name, price)
            VALUES (%s, %s)
        """
        cur.execute(sql, (data.indexName, data.price))
        conn.commit()
        
        return {"status": "success"}
    except Exception as e:
        if conn: conn.rollback()
        # 에러가 나면 n8n에서 정확한 이유를 볼 수 있게 500 에러를 뱉도록 수정
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            cur.close()
            conn.close()

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
async def ask_ai(question: str, refresh_news: bool = False):
    try:
        query_vec = get_embedding(question)
        stock_name, stock_code = find_stock(question)
        sentiment_context = "심리 데이터 없음"
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            # 글로벌 공포와 해당 종목 공포 두 가지를 다 가져옴
            cur.execute("""
                (SELECT * FROM market_sentiment WHERE stock_code = 'GLOBAL' ORDER BY id DESC LIMIT 1)
                UNION ALL
                (SELECT * FROM market_sentiment WHERE stock_code = %s ORDER BY id DESC LIMIT 1)
            """, (stock_code,))
            s_rows = cur.fetchall()
            if s_rows:
                sentiment_context = "\n".join([f"- {r['stock_code']} 심리점수: {r['score']}/100 ({r['reason']})" for r in s_rows])
            cur.close()
            conn.close()
        except: pass
        technical_info = get_technical_analysis(stock_name, stock_code) if stock_code else ""

        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # 💡 1. 한민님의 match_history 함수 호출 (과거 기록 검색)
        cur.execute("SELECT question, answer FROM match_history(%s::vector, %s, %s)", (str(query_vec), 0.5, 1))
        history_res = cur.fetchall()
        past_analysis = f"과거 질문: {history_res[0]['question']}\n과거 답변: {history_res[0]['answer']}" if history_res else ""

        # 💡 2. 질문 타입에 따른 뉴스 검색 분기 처리
        if stock_code:
            # [모드 A] 특정 종목을 물어본 경우: 벡터 유사도로 관련 뉴스 5개 검색
            cur.execute("SELECT title, content FROM match_news(%s::vector, %s, %s)", (str(query_vec), 0.3, 5))
            search_res = cur.fetchall()
        else:
            # [모드 B] 시장/테마를 물어본 경우: 무조건 가장 최근에 수집된 뉴스 15개 긁어오기
            cur.execute("SELECT title, content FROM news_data ORDER BY created_at DESC LIMIT 15")
            search_res = cur.fetchall()
        
        # 💡 3. 검색된 뉴스가 부족할 때 (네이버 크롤링 후 재검색)
        if refresh_news and len(search_res) < 2:
            fetch_naver_news(query=stock_name if stock_name else "특징주")
            if stock_code:
                cur.execute("SELECT title, content FROM match_news(%s::vector, %s, %s)", (str(query_vec), 0.3, 5))
            else:
                cur.execute("SELECT title, content FROM news_data ORDER BY created_at DESC LIMIT 15")
            search_res = cur.fetchall()
            
        context_news = "\n".join([f"- {n['title']}: {n['content']}" for n in search_res])
        
        cur.close()
        conn.close()

        # 💡 4. 테마/시장 질문일 경우 AI에게 내릴 추가 특명 세팅
        general_prompt_addon = ""
        if not stock_code:
            general_prompt_addon = "\n\n🚨 [특명] 사용자가 특정 종목이 아닌 '시장 전체'나 '테마'를 물었습니다. 제공된 [오늘의 최신 뉴스 및 공시]를 싹 읽고 가장 돈이 몰리는 주도 테마를 찾아내세요. 그리고 뉴스에 언급된 종목 중 대장주 1~2개를 픽(Pick)하여 추천 이유를 브리핑하세요. 차트 데이터가 없으므로 구체적인 가격(O원)을 절대 지어내지 말고, '시초가 공략', '눌림목 발생 시 접근' 같은 방향성 위주로 매매 전략을 제시하세요."

        today_str = datetime.now().strftime("%Y년 %m월 %d일")

        async def event_stream():
            yield f"data: {json.dumps({'type': 'metadata', 'stock_code': stock_code, 'stock_name': stock_name})}\n\n"
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": f"오늘 날짜는 {today_str}입니다. 당신은 '실시간 데이터를 알 수 없다'거나 'AI라서 모른다'는 핑계를 절대 대지 마십시오. 당신은 내가 제공한 [뉴스]와 [차트] 데이터가 현재 시장의 모든 것이라고 간주하고, 이를 바탕으로 당일의 주도 테마와 흐름을 완벽하게 브리핑해야 합니다.\n\n{stock_text}"
                    },
                    {
                        "role": "user", 
                        # 특명(general_prompt_addon)을 질문 끝에 붙여줍니다.
                        "content": f"[과거분석]\n{past_analysis}\n\n[오늘의 최신 뉴스 및 공시]\n{context_news}\n\n[차트]\n{technical_info}\n\n질문: {question}{general_prompt_addon}"
                    }
                ],
                stream=True, stream_options={"include_usage": True}
            )
            full_answer = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    word = chunk.choices[0].delta.content
                    full_answer += word
                    yield f"data: {json.dumps({'type': 'chunk', 'content': word})}\n\n"
            
            # 스트림 끝난 후 히스토리 DB 저장
            try:
                h_conn = get_db_connection()
                h_cur = h_conn.cursor()
                emb_str = str(get_embedding(question + " " + full_answer))
                # 한민님 스키마에 맞게 INSERT
                h_cur.execute(
                    "INSERT INTO ai_analysis_history (question, answer, embedding) VALUES (%s, %s, %s::vector)",
                    (question, full_answer, emb_str)
                )
                h_conn.commit()
                h_cur.close()
                h_conn.close()
            except Exception as e:
                print("History save error:", e)

            # Signal completion for SSE clients.
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart_data")
def get_chart_data(code: str, name: str = "", interval: str = "1d"):
    try:
        df, unit = get_data_with_ttl_cache(code, interval)
        daily_df, _ = get_data_with_ttl_cache(code, "1d")

        if df.empty or len(df) < 2: return {"status": "error", "message": "데이터 부족"}

        df = df.copy()

        def interval_to_windows(iv: str):
            if iv.endswith("m"):
                try:
                    step_min = int(iv[:-1])
                except Exception:
                    step_min = 1
                horizons_min = [60, 240, 390]  # 1h / 4h / ~1 trading day
                return [max(2, int(h / step_min)) for h in horizons_min]
            if iv == "1w":
                return [4, 12, 24]
            if iv == "1M":
                return [3, 6, 12]
            return [5, 20, 60]
        
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

        if not daily_df.empty and len(daily_df) >= 2:
            support_5 = float(daily_df['Low'].rolling(5).min().iloc[-1])
            resist_5 = float(daily_df['High'].rolling(5).max().iloc[-1])
            support_20 = float(daily_df['Low'].rolling(20).min().iloc[-1])
            resist_20 = float(daily_df['High'].rolling(20).max().iloc[-1])
            support_60 = float(daily_df['Low'].rolling(60).min().iloc[-1])
            resist_60 = float(daily_df['High'].rolling(60).max().iloc[-1])

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

        # Dynamic support/resistance based on requested interval (bars-based windows).
        sr_levels = []
        try:
            windows = interval_to_windows(interval)
            if "Low" in df.columns and "High" in df.columns:
                for bars in windows:
                    if len(df) >= bars:
                        tail = df.tail(bars)
                        sr_levels.append({
                            "bars": int(bars),
                            "support": float(tail["Low"].min()),
                            "resist": float(tail["High"].max()),
                        })
        except Exception:
            sr_levels = []

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

        primary_sr = sr_levels[0] if sr_levels else None

        return {
            "status": "success", "stock_code": code, "stock_name": name,
            "latest": {
                "as_of": latest["time"], "interval": interval, "unit": unit,
                "price": latest['close'], "rsi": safe_round(latest['rsi']), "volume_ratio": safe_round(vol_ratio), "ma5": safe_round(latest['ma5']),
                "support_5": safe_round(support_5), "resist_5": safe_round(resist_5),
                "support_20": safe_round(support_20), "resist_20": safe_round(resist_20),
                "support_60": safe_round(support_60), "resist_60": safe_round(resist_60),
                "pivot": safe_round(pivot), "s1": safe_round(s1), "s2": safe_round(s2), "r1": safe_round(r1), "r2": safe_round(r2),
                "support_dyn": safe_round(primary_sr["support"]) if primary_sr else None,
                "resist_dyn": safe_round(primary_sr["resist"]) if primary_sr else None,
                "sr_levels": [
                    {"bars": l["bars"], "support": safe_round(l["support"]), "resist": safe_round(l["resist"])}
                    for l in sr_levels
                ]
            },
            "history": res
        }
    except Exception as e: return {"status": "error", "message": str(e)}

@app.get("/fetch_dart")
def fetch_dart():
    conn = None
    try:
        dart_api_key = os.getenv("DART_API_KEY")
        if not dart_api_key:
            return {"status": "error", "message": "DART_API_KEY가 설정되지 않았습니다."}

        today = datetime.now().strftime("%Y%m%d")
        url = "https://opendart.fss.or.kr/api/list.json"
        
        conn = get_db_connection()
        cur = conn.cursor()

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
                    title = item['report_nm']
                    good_keywords = ["공급계약", "주식취득", "무상증자", "영업잠정실적", "타법인주식", "공개매수"]
                    
                    if any(k in title for k in good_keywords):
                        corp_name = item['corp_nm']
                        content = f"[{corp_name}] 전자공시: {title}"
                        link = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcp_no']}"
                        
                        cur.execute("SELECT id FROM news_data WHERE url = %s", (link,))
                        if not cur.fetchone():
                            emb_str = str(get_embedding(f"{corp_name} {title}"))
                            sql = """
                                INSERT INTO news_data (title, content, source, url, related_stocks, embedding)
                                VALUES (%s, %s, %s, %s, %s, %s::vector)
                            """
                            cur.execute(sql, (f"[공시] {corp_name} - {title}", content, "DART", link, None, emb_str))
                            conn.commit()

        return {"status": "success", "message": "DART 공시 수집 완료"}
    except Exception as e:
        if conn: conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            cur.close()
            conn.close()

@app.get("/fetch_kis_condition")
def fetch_kis_condition(seq: str = "1"): 
    conn = None
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
            "tr_id": "HHKST03900400",
            "custtype": "P"
        }
        params = {
            "user_id": os.getenv("KIS_USER_ID"), 
            "seq": seq, 
            "mac_address": os.getenv("MAC_ADDRESS").replace(":", "").replace("-", "").lower(),
            "bpass_chk_yn": "N",
            "clear_cmd_yn": "Y",
            "pblc_cmd_yn": "N",
            "out_func": "F",
            "ent_expt_cmd_yn": "N",
            "expt_cmd_yn": "N"
        }

        res = requests.get(url, headers=headers, params=params).json()
        
        if res.get('rt_cd') == '0':
            output2 = res.get('output2', [])
            
            if not output2:
                return {"status": "success", "message": "현재 조건식에 포착된 종목이 없습니다.", "data": []}
            
            detected_stocks = []
            for item in output2:
                stock_name = item['name']
                stock_code = item['code']
                detected_stocks.append(f"{stock_name}({stock_code})")
            
            related_stocks_str = ", ".join(detected_stocks)
            
            conn = get_db_connection()
            cur = conn.cursor()
            
            # 🚨 [추가된 핵심 로직] 가장 최근에 저장된 조건검색 결과 가져오기
            cur.execute("""
                SELECT related_stocks 
                FROM news_data 
                WHERE source = 'KIS_CONDITION' 
                ORDER BY id DESC LIMIT 1
            """)
            last_record = cur.fetchone()
            
            # 이전과 포착된 종목이 100% 똑같다면 저장하지 않고 건너뜀 (Skipped)
            if last_record and last_record[0] == related_stocks_str:
                return {"status": "skipped", "message": "종목 변동 없음 (DB 저장 생략)", "data": detected_stocks}

            # 🚨 종목에 변화가 생겼을 때만 아래 INSERT 실행
            title = f"[조건검색 포착] {len(detected_stocks)}종목 발굴"
            content = "포착 종목: " + related_stocks_str
            emb_str = str(get_embedding(content))
            
            sql = """
                INSERT INTO news_data (title, content, source, url, related_stocks, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
            """
            cur.execute(sql, (title, content, "KIS_CONDITION", f"kis_cond_{datetime.now().strftime('%Y%m%d%H%M')}", related_stocks_str, emb_str))
            conn.commit()
            
            return {"status": "success", "message": f"새로운 변동 발생! {len(detected_stocks)}개 종목 저장됨", "data": detected_stocks}
            
        return {"status": "error", "message": res.get('msg1')}
    except Exception as e:
        if conn: conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            cur.close()
            conn.close()

@app.get("/api/ranked_stocks")
async def get_ranked_stocks():
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT title, content, source, related_stocks 
            FROM news_data 
            WHERE created_at >= CURRENT_DATE 
            ORDER BY id DESC LIMIT 50
        """)
        res_data = cur.fetchall()

        if not res_data:
            return {"status": "error", "message": "DB에 데이터가 없습니다."}

        all_stocks = []
        for row in res_data:
            if row.get('related_stocks') and row['related_stocks'].strip() != 'NONE':
                stocks = [s.strip() for s in row['related_stocks'].split(',')]
                all_stocks.extend(stocks)
        
        if not all_stocks:
             return {"status": "error", "message": "데이터에서 종목 태그를 찾지 못했습니다."}

        top_5_tuples = Counter(all_stocks).most_common(5)
        target_stocks = [t[0] for t in top_5_tuples]

        stock_contexts = []
        recent_news_text = ""
        
        for stock_str in target_stocks:
            if "(" in stock_str and ")" in stock_str:
                name = stock_str.split("(")[0]
                code = stock_str.split("(")[1].replace(")", "")
            else:
                name, code = find_stock(stock_str)
            
            if code:
                tech_info = get_technical_analysis(name, code)
                stock_contexts.append(f"[{name}({code})]\n{tech_info}")
                
                relevant_news = [r for r in res_data if r.get('related_stocks') and name in r['related_stocks']][:2]
                for rn in relevant_news:
                    recent_news_text += f"- [{rn['source']}] {rn['title']}\n"

        if not stock_contexts:
             return {"status": "error", "message": "종목 기술적 지표를 불러오지 못했습니다."}

        combined_tech = "\n\n".join(stock_contexts)

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
            response_format={ "type": "json_object" }
        )

        raw_json = score_res.choices[0].message.content
        parsed_data = json.loads(raw_json)
        stocks_list = parsed_data.get("stocks", [])
        sorted_stocks = sorted(stocks_list, key=lambda x: x.get('score', 0), reverse=True)

        return {"status": "success", "data": sorted_stocks}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            cur.close()
            conn.close()

# ==========================================
# 7. 심리 분석 및 공포 지수 (추가 기능)
# ==========================================

class SentimentData(BaseModel):
    stock_code: str
    score: float
    reason: str

@app.get("/fetch_market_sentiment")
def fetch_market_sentiment(code: str = "GLOBAL", name: str = ""):
    """
    GLOBAL이면 VIX 수집, 특정 코드를 넣으면 최근 뉴스 + 차트 데이터 기반으로 GPT가 심리 점수 산출
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        score = 50.0
        reason = "데이터 부족"
        vix_val = None

        if code == "GLOBAL":
            # 1. 매크로 공포 지수 (VIX) 가져오기
            vix = yf.Ticker("^VIX").history(period="1d", interval="1m")
            if not vix.empty:
                vix_val = float(vix['Close'].iloc[-1])
                # VIX 30 이상이면 공포(20점), 15 이하면 환희(80점)로 변환
                score = max(0, min(100, 100 - (vix_val * 2))) 
                reason = f"글로벌 VIX 지수 현재 {vix_val:.2f} 기록 중"
        else:
            # 2. 개별 종목 심리 (뉴스 + 차트 데이터 융합 분석)
            # 💡 [핵심 수정 1] 이름(name) 파라미터를 추가하여 뉴스 '제목(title)'에서도 종목명을 검색하도록 SQL 확장
            if name:
                cur.execute(
                    "SELECT title FROM news_data WHERE related_stocks LIKE %s OR title LIKE %s ORDER BY id DESC LIMIT 10", 
                    (f"%{code}%", f"%{name}%")
                )
            else:
                cur.execute(
                    "SELECT title FROM news_data WHERE related_stocks LIKE %s ORDER BY id DESC LIMIT 10", 
                    (f"%{code}%",)
                )
            
            news_rows = cur.fetchall()
            titles = [r['title'] for r in news_rows]
            
            if titles:
                # 💡 [핵심 수정 2] 기존에 만들어둔 차트 지표 조회 함수 활용
                tech_info = get_technical_analysis(name, code) if code else "차트 데이터 없음"

                # 💡 [핵심 수정 3] 뉴스와 차트 지표를 모두 융합하여 더욱 정교한 투자 심리를 분석하도록 프롬프트 고도화
                analysis_prompt = f"""
                당신은 냉혹한 주식 투자 심리 분석가입니다.
                아래의 [최근 뉴스]와 [현재 기술적 지표]를 종합적으로 판단하여 이 종목의 현재 투자자 심리(FOMO, 패닉셀 등)를 0~100점으로 수치화하세요.
                0은 극단적 공포/패닉, 50은 중립/관망, 100은 극단적 환희/광기입니다.
                - 뉴스 호재가 있어도 차트(RSI 과매수, 저항선 부근)가 과열 상태면 점수를 보수적으로 잡고,
                - 악재가 있어도 바닥 지지 및 거래량 폭발이 있으면 반등 기대감으로 점수를 올리세요.

                [최근 뉴스]
                {titles}

                [기술적 지표]
                {tech_info}

                반드시 숫자와 한줄 이유만 JSON으로 응답하세요.
                포맷 예시: {{"score": 85, "reason": "호재 뉴스가 있으나 RSI 75 초과 및 20일선 이격도 과대로 단기 차익매물 출회 우려"}}
                """
                
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    response_format={"type": "json_object"}
                )
                parsed = json.loads(res.choices[0].message.content)
                score = parsed['score']
                reason = parsed['reason']

        # 💡 [핵심 수정 4] 뉴스가 없어서 50점이 나온 경우는 DB에 저장하지 않고 스킵 (DB 스팸 방지)
        if reason != "데이터 부족":
            cur.execute(
                "INSERT INTO market_sentiment (stock_code, score, vix_value, reason) VALUES (%s, %s, %s, %s)",
                (code, score, vix_val, reason)
            )
            conn.commit()
            
        return {"status": "success", "score": score, "reason": reason, "vix": vix_val}

    except Exception as e:
        if conn: conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        if conn:
            cur.close()
            conn.close()

@app.get("/debug_mac")
def debug_mac():
    raw = os.getenv("MAC_ADDRESS", "없음")
    cleaned = raw.replace(":", "").replace("-", "")
    return {
        "raw": raw,
        "cleaned": cleaned,
        "length": len(cleaned)
    }

@app.get("/")
def serve_ui(): return FileResponse("index.html", headers={"Cache-Control": "no-store"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
