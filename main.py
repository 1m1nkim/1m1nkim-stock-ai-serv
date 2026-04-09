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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from postgrest import SyncPostgrestClient
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

# 1. 초기 설정 및 클라이언트
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
당신은 사전적 정의를 읊는 봇이 아니라, 테마의 확장성과 돈의 흐름을 읽고 51%의 승률 자리를 찾는 실전 트레이더입니다.
아래의 [실전 관점]을 철저히 1순위 기준으로 삼아 상황을 해석하고 답변하세요.

[1. 가치 지표의 실전 해석 (심리 읽기)]
- PER: '현재 가치'가 아닌 '미래 기대치'다. 낮다고 무조건 싼 게 아니라 성장성이 죽었을 수 있고, 높다고 나쁜 게 아니라 시장이 미래에 베팅 중인 것이다.
- PBR: 1 이하라고 싼 게 아니다. 시장의 신뢰를 잃었다고 의심부터 해라.
- ROE: 단순히 돈을 잘 버는 게 아니라, 자본 굴리는 '효율'이 좋다는 뜻이다. 부채가 높으면 위험하다.
- EPS: 주가 방향의 연료다. 증가하면 성장, 감소하면 기대가 꺾인 것이다.
- 배당수익률: 고배당이라고 무조건 안전한 게 아니다. 더 이상 성장할 곳이 없어 성장을 포기한 기업일 수도 있다.
- 시가총액: 돈은 항상 작은 데서 큰 데로 이동한다. 대형주는 안정적이지만, 소형주는 테마와 수급이 붙으면 변동성과 기회가 크다.
- 거래량: 가격보다 먼저 움직이는 핵심 신호다. 재료 없이 터지면 세력 개입, 거래량 없는 상승은 지속성이 없다.

[2. 기술적 지표의 본질 (추세와 공포)]
- 이동평균선(MA): '평균 매수단가'다. 20일선 위는 단기 강세, 60일선 위는 중기 추세 유지다. 캔들 하나에 집착하지 말고 이평선이 만드는 '숲(추세)'을 봐라.
- 골든/데드크로스: 예측 신호가 아니라 이미 벌어진 '결과'다. 이미 오른 뒤에 나오는 골든크로스를 추격 매수하면 물린다.
- RSI: 30 이하는 단순히 '싼 자리'가 아니라 극도의 '공포 구간'이다. 공포에 질릴 때가 아니라, '공포가 끝나는 자리(반등)'에서 매수해야 한다.
- 볼린저밴드: 밴드가 좁아졌다가 벌어지는 수축기가 바로 '큰 움직임의 전조'다. 상/하단 터치만으로 맹신하지 마라.
- MACD: 후행 지표다. 절대적 기준이 아닌 보조 참고용으로만 써라.

[3. 실전 투자 철학 및 시장 대응 (마인드셋)]
- 테마의 확장성: 시장은 '스토리'로 움직인다. (예: 전쟁 → 에너지 → 방산 → 곡물 → 재건). 현재 핫한 테마의 다음 '확산될 파생 테마'를 미리 선점해라.
- 가격보다 타이밍: 좋은 기업이라고 무조건 오르지 않는다(선반영). 주가는 기업의 가치뿐만 아니라 '기대감'과 '돈의 흐름(수급)'이 만든다.
- 3가지 장세 대응법:
  1) 하락장(거래대금 감소): 철저한 테마 단기 매매로 대응.
  2) 급락장(단기 이벤트/전쟁/금리): 분할 매수로 반등을 노린다.
  3) 폭락장(금융 시스템 붕괴): 쥐고 있던 현금으로 우량주/지수를 장기 매집한다. 기회는 하락장에서만 온다.
- 매매 절대 원칙: 감정 매매, 고점 추격, 몰빵은 절대 금지다. 주식은 100%가 없는 51%의 확률 싸움이다. 철저히 분할매수/분할매도로 심리적 우위를 점해라.

[4. 종목 분석 프로세스]
사용자가 종목을 물어보면 반드시 다음 순서로 속으로 사고한 뒤 결론을 내라:
1) 스토리: 이 기업이 왜 움직이는가?
2) 테마 위치: 현재 테마의 초기/중기/끝물 중 어디인가?
3) 수급/거래량: 거래량이 들어왔고, 누가 사고 있는가?
4) 차트 자리: 현재 위치가 고점/눌림목/바닥 중 어디인가?
5) 결론 도출: 51%의 승률이 나오는 자리인가? 애매하면 '관망/대기', 승산이 있으면 '분할 진입'을 명확히 제시하라.
6) 매매 타점 (필수 출력): 차트 지표로 전달받은 '매수/매도 신호', '목표가', '손절가'를 반드시 그대로 브리핑해라. 만약 '매수 타점 아님'이라면 왜 타점이 아닌지 다시 한번 뼈 때리게 조언해라.

[5. 일정/수급 선취매 관점]
- 지수 편입 선취매: 코스닥150 등 편입 예상 종목은 발표 3~4개월 전부터 패시브 수급이 선반영된다. 바닥권 선취매 후 대기가 핵심이다.
- 거시 일정 역산: 금리 결정, 인베스터 데이, 정책 발표 전에 수혜 테마를 미리 역산해서 진입해라. 뉴스 나오고 사면 늦다.
"""

knowledge_text = """
[참고 문서 - 투자 개념 상세 설명]
1. 기본 가치 지표 — 숫자가 아니라 "왜"를 봐라
PER (주가수익비율)
주가 ÷ EPS. 낮을수록 저평가처럼 보이지만 그게 전부가 아니다.
PER이 낮은 이유가 성장성이 없어서일 수도 있고, PER이 높아도 시장이 미래 성장에 베팅 중인 거라면 오히려 기회일 수 있다.
핵심은 이거다. PER은 현재 가치가 아니라 "미래 기대가 얼마나 반영됐는지"를 보는 지표다. 숫자 하나만 보지 말고 왜 그 PER이 나왔는지 맥락을 봐라.
PBR (주가순자산비율)
주가 ÷ 순자산. 1 미만이면 자산보다 싸게 팔린다는 뜻이지만, 싸다고 좋은 게 아니다.
PBR 1 미만은 "시장이 이 기업을 신뢰하지 않는다"는 신호일 수 있다. 싸 보이는 이유를 먼저 의심해라. 하방 경직성은 있지만 오르지 않는 이유도 반드시 있다.
ROE (자기자본이익률)
자기자본으로 얼마나 이익을 뽑아냈냐. 높을수록 자본 효율이 좋은 기업이다.
단 ROE가 높은데 부채도 높으면 위험하다. ROE 높고 재무 안정적이면 그게 진짜 우량이다. 레버리지로 만들어진 ROE인지 체력으로 만들어진 ROE인지 구분해라.
EPS (주당순이익)
당기순이익 ÷ 발행주식수. 내가 주식 한 주로 회사가 얼마를 벌어줬냐는 거다.
EPS가 분기마다 꾸준히 늘어나는 기업이 장기 우상향한다. EPS 증가는 기업 성장 신호, EPS 감소는 시장 기대가 꺾이는 신호다. EPS는 주가 방향의 연료다.
배당수익률
주가 대비 배당금 비율. 높으면 들고만 있어도 현금이 들어와서 하락장에서 하방 경직성을 제공한다.
단 고배당이라고 무조건 좋은 게 아니다. 성장에 쓸 곳이 없어서 배당으로 돌리는 성장 포기 기업일 수도 있다. 배당수익률은 방어주 판단용이지 성장주에 들이대는 지표가 아니다.
시가총액
주가 × 발행주식수. 회사의 몸값이다.
대형주는 안정적이지만 느리고, 소형주는 변동성이 크지만 기회도 크다. 돈은 항상 작은 데서 큰 데로 이동한다. 테마가 붙으면 시총 작은 종목이 먼저 크게 움직인다.
거래량
하루 동안 거래된 주식 수. 가격보다 거래량이 먼저 움직인다.
거래량 없이 주가만 오르면 지속성이 낮다. 거래량이 평소보다 크게 터지면 세력 개입 가능성이 있다. 바닥권에서 거래량이 급증하면 세력 매집 및 추세 전환의 강력한 신호다.
52주 신고가 / 신저가
최근 1년의 최고가와 최저가. 신고가 돌파면 위에 매물이 없어서 추가 상승 여력이 생긴다. 신저가 근처면 바닥인지 아닌지 펀더멘탈 확인이 필요한 자리다.

2. 기술적 분석 — 차트는 심리의 기록이다
이동평균선 (MA)
5일, 20일, 60일, 120일선이 주로 쓰인다. 이평선은 평균 매수단가라고 생각해라.
20일선 위면 단기 강세, 60일선 위면 중기 추세 유지 중이다. 주가가 모든 이평선 위에 있으면 강세 추세다.
골든크로스 / 데드크로스
단기선이 장기선을 위로 뚫으면 골든크로스(상승 신호), 아래로 꿰뚫으면 데드크로스(하락 신호)다.
근데 이게 신호가 아니라 결과라는 걸 알아야 한다. 이미 오른 뒤에 골든크로스가 나온다. 초보는 여기서 물린다.
RSI (상대강도지수)
0~100 사이. 70 이상이면 과매수로 단기 고점 주의, 30 이하면 과매도로 반등 기회 구간이다.
RSI 30 이하라고 무조건 사는 게 아니다. 공포 구간이라는 뜻이지 바닥이라는 뜻이 아니다. 공포에서 사는 게 아니라 "공포가 끝나는 자리"에서 사는 거다. 형이 코코레 타이밍으로 코스닥 RSI 30 얘기하는 게 바로 이 논리다.
볼린저밴드
이동평균선 위아래로 표준편차 2배 밴드. 상단이라고 무조건 매도, 하단이라고 무조건 매수가 아니다.
밴드가 좁아지면(수축) 큰 움직임의 전조다. 방향은 모르지만 에너지가 쌓이고 있다는 신호다.
MACD
단기와 장기 이평선의 차이를 이용한 후행 지표다. 늦다. 시그널선 교차 시 참고는 하되 기준으로 삼지 마라. RSI랑 같이 보면 정확도가 올라간다.
거래량 이동평균
평균 거래량보다 크게 터지면 강한 매수 또는 매도 신호다. 세력이 물량 모으거나 털 때 거래량이 먼저 반응한다. 캔들이랑 거래량 이 두 가지는 항상 같이 봐라.

3. 시장 테마 분석 관점 — 플스포 뷰
테마 확장의 법칙
전쟁이나 전염병 같은 강력한 거시 테마는 단발성으로 끝나지 않는다.
코로나 하나로 진단 → 치료제 → 마스크 → 배달 → 교육까지 테마가 퍼졌다. 전쟁도 에너지 → 방산 → 곡물/사료 → 비료 → 해운 → 재건으로 확산됐다.
1파 상승을 놓쳤다면 다음에 확산될 테마를 역산해서 선매집하는 게 관점이다. 지금 테마가 아니라 다음 테마를 먼저 봐라.
데이터 패권 기업의 본질
팔란티어나 테슬라 같은 기업은 단순 소프트웨어나 제조업이 아니다. 사용자의 행동 데이터를 독점하여 알고리즘을 고도화하는 데이터 플랫폼이다.
이런 기업은 사용자가 늘수록 데이터가 쌓이고, 데이터가 쌓일수록 서비스가 정밀해지고, 정밀해질수록 더 많은 사용자가 모이는 선순환 구조를 가진다. 단기 노이즈에 흔들리지 말고 진입장벽과 데이터 축적 속도라는 본질을 보고 장기 동행해라.
하락장 / 급락장 / 폭락장 생존 전략
거래대금 급감을 통해 하락장을 조기에 캐치해야 한다. 매일 거래대금 체크하고 큰 돈의 흐름을 느껴라.
급락장(전쟁, 금리 이슈 등 단기 악재)은 빠르게 회복된다. 펀더멘탈 멀쩡한데 지수 때문에 밀린 종목을 현금으로 분할 매수하는 게 전략이다. 코스닥 RSI 30 이탈 시 코코레 매수도 이 논리다.
폭락장(시스템 붕괴)은 현금이 먼저다. 회복이 느리지만 시대가 지날수록 회복 기간은 짧아지고 있다. 오히려 장기 투자 종목을 더 모을 절호의 기회다.
세 가지 공통점은 현금, 회복, 테마주다.

4. 일정 및 수급 매매 관점 — 겜보이 뷰
지수 편입 선취매 전략
코스닥 150 같은 주요 지수에 신규 편입이 예상되는 종목은 패시브 펀드의 기계적 매수세가 강제 유입된다. 펀드매니저의 의지나 밸류에이션과 무관하게 무조건 사야 하는 구조다.
시장은 이를 미리 선반영한다. 통계적으로 편입 발표 기준 3~4개월 전부터 수익률이 좋아진다. 바닥권에서 먼저 선취매하고 기다리는 게 가장 안전한 수익 모델이다.
거시 일정의 나비효과
미국 연준 금리 결정, 핵심 기업의 인베스터 데이, 정부 정책 발표, 글로벌 컨퍼런스 일정은 관련 섹터에 강한 모멘텀을 준다.
뉴스가 나오기 전에 예정된 일정을 파악하고, 수혜를 받을 테마를 역산해서 미리 들어가야 한다. 뉴스 나오고 사면 이미 늦다.

5. 실전 적용 프레임 — 종목 볼 때 순서

이 기업 왜 움직이냐 (스토리와 테마 파악)
지금 테마 어디냐 (초기 / 중기 / 끝물)
거래량 들어왔냐
수급 누가 사냐 (외인 / 기관 / 세력)
지금 자리가 어디냐 (고점 / 눌림 / 바닥)
확률 51% 넘냐 → 넘으면 분할 매수, 애매하면 대기, 확신 없으면 패스

절대 원칙: 고점 추격 금지 / 몰빵 금지 / 감정 매매 금지 / 현금 비중 50% 유지
"""

# 2. 데이터 모델
class NewsData(BaseModel):
    title: str
    content: str
    source: str
    url: str

class StockQuery(BaseModel):
    question: str

# KRX 종목 리스트 로딩 (서버 시작 시 1회)
print("✅ KRX 종목 리스트 로딩 중...")
krx = fdr.StockListing('KRX')

# 3. 핵심 유틸리티 함수

def get_embedding(text: str):
    safe_text = text[:1000] if text else ""
    response = client.embeddings.create(
        input=safe_text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def find_stock(question: str):
    """질문에서 종목명을 찾아 이름과 코드를 반환"""
    
    # 1. 먼저 KRX에서 한국 종목 검색
    q_no_space = question.replace(" ", "")
    matched = krx[krx['Name'].apply(lambda n: n.replace(" ", "") in q_no_space)]
    if not matched.empty:
        best = matched.loc[matched['Name'].str.len().idxmax()]
        return best['Name'], best['Code']
    
    # 2. KRX에 없으면 GPT로 미국 티커 추출
    try:
        ticker_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "사용자 질문에서 미국 주식 티커 심볼만 추출해서 대문자로 반환하세요. 티커만 반환하고 다른 말은 하지 마세요. 없으면 NONE"
                },
                {"role": "user", "content": question}
            ],
            max_tokens=10
        )
        ticker = ticker_response.choices[0].message.content.strip()
        
        if ticker != "NONE" and len(ticker) <= 5:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName")
            if name:
                print(f"✅ 미국 종목 인식: {name} ({ticker})")
                return name, f"US:{ticker}"
                
    except Exception as e:
        print(f"미국 종목 검색 실패: {e}")
    
    return None, None

def get_technical_analysis(name: str, code: str) -> str:
    try:
        if code.startswith("US:"):
            ticker = code.replace("US:", "")
            df = yf.download(ticker, period="6mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            unit = "달러"
        else:
            df = fdr.DataReader(code).tail(500)
            unit = "원"

        if df.empty or len(df) < 60:
            return f"[{name} 데이터 부족]"

        close  = df['Close']
        volume = df['Volume']
        high   = df['High']
        low    = df['Low']

        price = round(float(close.iloc[-1]), 2)
        ma5   = round(float(close.rolling(5).mean().iloc[-1]), 2)
        ma20  = round(float(close.rolling(20).mean().iloc[-1]), 2)
        ma60  = round(float(close.rolling(60).mean().iloc[-1]), 2)

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi   = float((100 - 100 / (1 + gain / loss)).iloc[-1])

        # MACD
        ema12  = close.ewm(span=12).mean()
        ema26  = close.ewm(span=26).mean()
        macd   = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_val   = float(macd.iloc[-1])
        signal_val = float(signal.iloc[-1])

        # 볼린저밴드
        bb_mid   = close.rolling(20).mean()
        bb_std   = close.rolling(20).std()
        bb_upper = round(float((bb_mid + 2 * bb_std).iloc[-1]), 2)
        bb_lower = round(float((bb_mid - 2 * bb_std).iloc[-1]), 2)
        bb_width = round(float(((bb_upper - bb_lower) / float(bb_mid.iloc[-1])) * 100), 2)

        # 거래량 분석
        vol_ma20     = volume.rolling(20).mean()
        vol_ratio    = round(float(volume.iloc[-1] / vol_ma20.iloc[-1]), 2)  # 평균 대비 배수
        vol_trend    = "증가" if volume.iloc[-1] > vol_ma20.iloc[-1] else "감소"

        # 지지/저항 (20일 고점/저점)
        resist = round(float(high.rolling(20).max().iloc[-1]), 2)
        support = round(float(low.rolling(20).min().iloc[-1]), 2)

        # 타점 판단
        entry_signal = []
        target_price = None
        stop_loss    = None

        # 매수 타점 조건
        if rsi <= 35 and macd_val > signal_val:
            entry_signal.append("RSI 과매도 + MACD 반등")
        if price <= ma20 * 1.02 and price >= ma20 * 0.98:
            entry_signal.append("20일선 눌림목")
        if vol_ratio >= 1.5 and macd_val > signal_val:
            entry_signal.append("거래량 급증 + MACD 매수")
        if price <= bb_lower * 1.02:
            entry_signal.append("볼린저밴드 하단 반등 구간")

        # 매도 타점 조건
        sell_signal = []
        if rsi >= 70:
            sell_signal.append("RSI 과매수 구간")
        if price >= bb_upper * 0.98:
            sell_signal.append("볼린저밴드 상단 근접")
        if price >= resist * 0.98:
            sell_signal.append("20일 저항선 근접")

        # 목표가 / 손절가 계산
        if entry_signal:
            target_price = round(price * 1.08, 2)   # 8% 목표
            stop_loss    = round(price * 0.95, 2)    # 5% 손절

        # 🔥 에러 해결 부분: None 방어 로직 추가
        target_str = f"{target_price:,}{unit}" if target_price is not None else "매수 타점 아님"
        stop_str   = f"{stop_loss:,}{unit}" if stop_loss is not None else "매수 타점 아님"

        return (
            f"[{name} 기술적 지표]\n"
            f"- 현재가: {price:,}{unit}\n"
            f"- 이동평균: 5일({ma5:,}) / 20일({ma20:,}) / 60일({ma60:,})\n"
            f"- RSI: {rsi:.1f} ({'과매수' if rsi>=70 else '과매도' if rsi<=30 else '중립'})\n"
            f"- MACD: {'매수신호' if macd_val > signal_val else '매도신호'}\n"
            f"- 볼린저밴드: 상단({bb_upper:,}) / 하단({bb_lower:,}) / 밴드폭({bb_width}%)\n"
            f"- 거래량: 평균 대비 {vol_ratio}배 ({vol_trend})\n"
            f"- 지지선: {support:,}{unit} / 저항선: {resist:,}{unit}\n"
            f"\n[타점 분석]\n"
            f"- 매수 신호: {', '.join(entry_signal) if entry_signal else '현재 없음'}\n"
            f"- 매도 신호: {', '.join(sell_signal) if sell_signal else '현재 없음'}\n"
            f"- 목표가: {target_str}\n"
            f"- 손절가: {stop_str}\n"
        )

    except Exception as e:
        return f"기술적 분석 실패: {e}"

# 4. 엔드포인트

@app.post("/collect")
def collect_news(data: NewsData):
    try:
        check_duplicate = supabase.table("news_data").select("id").eq("url", data.url).execute()
        if len(check_duplicate.data) > 0:
            return {"status": "skipped", "message": "이미 존재하는 데이터입니다."}

        vector = get_embedding(f"{data.title}\n{data.content}")
        insert_data = {**data.dict(), "embedding": vector}
        supabase.table("news_data").insert(insert_data).execute()
        return {"status": "success", "message": "저장 완료"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/fetch_naver_news")
def fetch_naver_news(query: str = "증시"):
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    url = f"https://openapi.naver.com/v1/search/news.json?query={query}&display=10&sort=date"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    
    response = requests.get(url, headers=headers)
    items = response.json().get("items", [])
    for item in items:
        collect_news(NewsData(
            title=item['title'].replace("<b>","").replace("</b>",""),
            content=item['description'].replace("<b>","").replace("</b>",""),
            source="Naver",
            url=item['link']
        ))
        time.sleep(0.5)
    return {"status": "success", "count": len(items)}

@app.get("/ask_ai")
def ask_ai(question: str):
    try:
        # [1단계] 질문 임베딩 생성
        query_vec = get_embedding(question)

        # [2단계] 종목 식별 및 차트 데이터 준비
        stock_name, stock_code = find_stock(question)
        print(f"🔍 종목 인식 결과: name={stock_name}, code={stock_code}")
        
        technical_info = get_technical_analysis(stock_name, stock_code) if stock_code else ""
        print(f"📊 차트 데이터: {technical_info[:100] if technical_info else '없음'}")

        # [3단계] 과거 AI 분석 히스토리 검색 (RAG 1: Memory)
        history_res = supabase.rpc("match_history", {
            "query_embedding": query_vec,
            "match_threshold": 0.5,
            "match_count": 1
        }).execute()

        past_analysis = ""
        if history_res.data:
            past_analysis = (
                f"과거 질문: {history_res.data[0]['question']}\n"
                f"과거 답변: {history_res.data[0]['answer']}"
            )

        # [4단계] 최신 뉴스 검색 (RAG 2: News)
        search_res = supabase.rpc("match_news", {
            "query_embedding": query_vec,
            "match_threshold": 0.3,
            "match_count": 5
        }).execute()

        if len(search_res.data) < 2:
            fetch_naver_news(query=stock_name if stock_name else question)
            search_res = supabase.rpc("match_news", {
                "query_embedding": query_vec,
                "match_threshold": 0.3,
                "match_count": 5
            }).execute()

        context_news = "\n".join([
            f"- {n['title']}: {n['content']}" 
            for n in search_res.data
        ])

        # [5단계] 답변 생성
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": stock_text  # 행동 지침 고정
                },
                {
                    "role": "user",
                    "content": f"""
[투자 개념 참고 문서]
{knowledge_text}

[과거 분석 내용 (참고용)]
{past_analysis if past_analysis else "관련된 과거 분석 기록 없음."}

[최신 뉴스 및 기술적 지표]
뉴스: {context_news}
차트 지표: {technical_info}

[수행 지침]
1. 과거 분석이 있다면 현재와 비교해 전망이 개선/악화됐는지 언급하세요.
2. 실전 관점(51% 확률, 분할매수)을 기준으로 해석하세요.
3. 종목 분석 5단계(스토리→테마위치→수급→차트자리→결론) 순서로 결론을 내세요.
4. 과거와 현재의 괴리를 찾아 논리적 결론을 도출하세요.

사용자 질문: {question}
                    """
                }
            ]
        )
        final_answer = response.choices[0].message.content

        # [6단계] 답변 저장
        answer_vec = get_embedding(question + " " + final_answer)
        supabase.table("ai_analysis_history").insert({
            "question": question,
            "answer": final_answer,
            "embedding": answer_vec
        }).execute()

        return {
            "answer": final_answer,
            "has_past_reference": bool(past_analysis),
            "sources": [n['url'] for n in search_res.data],
            "stock_code": stock_code,
            "stock_name": stock_name
        }

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart_data")
def get_chart_data(code: str, name: str = ""):
    try:
        import numpy as np
        
        if code.startswith("US:"):
            ticker = code.replace("US:", "")
            df = yf.download(ticker, period="6mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
        else:
            df = fdr.DataReader(code).tail(120)

        if df.empty or len(df) < 20:
            return {"status": "error", "message": "데이터 부족"}

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

        df = df.dropna(subset=['ma20']).tail(400)

        res = []
        for idx, row in df.iterrows():
            signal = None
            price = float(row['Close'])
            # 1. 밴드 상단/하단
            if not pd.isna(row['bb_lower']) and price <= float(row['bb_lower']) * 1.01:
                signal = 'BB하단 지지(눌림)'
            elif not pd.isna(row['bb_upper']) and price >= float(row['bb_upper']) * 0.99:
                signal = 'BB상단 저항'
            # 2. 60일선 (장기)
            elif not pd.isna(row['ma60']) and price <= float(row['ma60']) * 1.015 and price >= float(row['ma60']) * 0.985:
                # 돌파인지 눌림인지
                if row['ma5'] > row['ma60']:
                    signal = '60일선 안착'
                else:
                    signal = '60일선 지지/저항'
            # 3. 20일선 (단기)
            elif not pd.isna(row['ma20']) and price <= float(row['ma20']) * 1.01 and price >= float(row['ma20']) * 0.99:
                signal = '20일선 눌림'
            
            res.append({
                "time": str(idx.date()) if hasattr(idx, 'date') else str(idx).split('T')[0],
                "open": float(row['Open']) if 'Open' in row else price,
                "high": float(row['High']) if 'High' in row else price,
                "low": float(row['Low']) if 'Low' in row else price,
                "close": price,
                "ma5": float(row['ma5']) if not pd.isna(row['ma5']) else None,
                "ma20": float(row['ma20']) if not pd.isna(row['ma20']) else None,
                "ma60": float(row['ma60']) if not pd.isna(row['ma60']) else None,
                "bb_upper": float(row['bb_upper']) if not pd.isna(row['bb_upper']) else None,
                "bb_lower": float(row['bb_lower']) if not pd.isna(row['bb_lower']) else None,
                "rsi": float(row['rsi']) if not pd.isna(row['rsi']) else None,
                "volume": float(row['Volume']),
                "signal": signal
            })

        latest = res[-1]
        
        # NaN safe conversion for latest data
        vol_ma20 = df['Volume'].rolling(20).mean()
        recent_vol_ma = vol_ma20.iloc[-1]
        vol_ratio = float(df['Volume'].iloc[-1] / recent_vol_ma) if not pd.isna(recent_vol_ma) and recent_vol_ma > 0 else 1.0

        def safe_round(val, decimals=1):
            if val is None or pd.isna(val):
                return None
            return round(val, decimals)

        return {
            "status": "success",
            "stock_code": code,
            "stock_name": name,
            "latest": {
                "price": latest['close'],
                "rsi": safe_round(latest['rsi'], 1) if latest['rsi'] else 50,
                "volume_ratio": safe_round(vol_ratio, 2),
                "ma5": safe_round(latest['ma5'], 2)
            },
            "history": res
        }
    except Exception as e:
        print(f"Chart data error: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/")
def serve_ui():
    return FileResponse("index.html")