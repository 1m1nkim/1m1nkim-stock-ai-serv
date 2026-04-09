import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { useStore } from './store';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

function ChatPanel() {
  const [messages, setMessages] = useState([
    {
      type: 'ai',
      text: '안녕하세요. 종목에 대한 실전 분석을 도와드리는 **Quant AI**입니다.\n궁금하신 종목명이나 질문을 편하게 남겨주세요!\n\n예: "한전산업은 어때?", "삼성전자 분석해줘"'
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const setStockInfo = useStore(state => state.setStockInfo);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { type: 'user', text: userMsg }]);
    setInput('');
    setLoading(true);

    try {
      // url path for proxy or direct access
      const url = `/ask_ai?question=${encodeURIComponent(userMsg)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error('서버 응답 오류');
      const data = await res.json();
      
      setMessages(prev => [...prev, {
        type: 'ai',
        text: data.answer,
        sources: data.sources
      }]);

      if (data.stock_code) {
        setStockInfo(data.stock_code, data.stock_name, null);
      }

    } catch (err) {
      setMessages(prev => [...prev, { type: 'ai', text: `오류가 발생했습니다: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-panel backdrop-blur-md rounded-2xl border border-border-color shadow-2xl p-6">
      <div className="mb-6 flex-shrink-0">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-accent to-indigo-400 bg-clip-text text-transparent">
          Quant AI Analyst
        </h1>
        <p className="text-sm text-text-muted mt-1">실전 트레이딩 관점의 종목 분석을 제공합니다</p>
      </div>

      <div className="flex-1 overflow-y-auto pr-2 space-y-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl p-4 shadow-sm ${msg.type === 'user' ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-br-sm' : 'bg-slate-700/50 border border-border-color rounded-bl-sm'}`}>
              <div className="prose prose-invert max-w-none text-[0.95rem] leading-relaxed">
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              </div>
              {msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-3 border-t border-border-color/50 text-xs text-text-muted">
                  <span className="block font-semibold mb-2">참고 문서</span>
                  <div className="flex flex-wrap gap-2">
                    {[...new Set(msg.sources)].map((src, idx) => {
                      let domain = "Link";
                      try { domain = new URL(src).hostname.replace('www.', ''); } catch (e) {}
                      return (
                        <a key={idx} href={src} target="_blank" rel="noreferrer" className="inline-block px-2 py-1 bg-accent/10 text-accent rounded-lg hover:bg-accent/20 transition-colors">
                          📄 {domain}
                        </a>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-700/50 border border-border-color rounded-2xl rounded-bl-sm p-5 space-y-3 w-2/3 max-w-sm">
              <div className="h-4 bg-slate-600/50 rounded animate-pulse w-3/4"></div>
              <div className="h-4 bg-slate-600/50 rounded animate-pulse w-full"></div>
              <div className="h-4 bg-slate-600/50 rounded animate-pulse w-5/6"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="mt-4 flex-shrink-0 flex gap-3 pt-2">
        <input 
          type="text" 
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="종목명이나 질문을 입력해주세요..."
          className="flex-1 bg-slate-800/60 border border-border-color rounded-xl px-4 py-3 text-white placeholder-text-muted outline-none focus:border-accent focus:ring-1 focus:ring-accent transition-all"
        />
        <button 
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="bg-gradient-to-r from-blue-500 to-blue-600 text-white font-semibold px-8 rounded-xl hover:from-blue-400 hover:to-blue-500 disabled:opacity-50 transition-all shadow-lg"
        >
          전송
        </button>
      </div>
    </div>
  );
}

function DashboardPanel() {
  const { stockCode, stockName } = useStore();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Toggles
  const [showMA5, setShowMA5] = useState(true);
  const [showMA20, setShowMA20] = useState(true);
  const [showMA60, setShowMA60] = useState(false);

  useEffect(() => {
    if (!stockCode) return;
    
    let isMounted = true;
    const fetchChart = async () => {
      setLoading(true);
      try {
        const url = `/api/chart_data?code=${encodeURIComponent(stockCode)}&name=${encodeURIComponent(stockName || '')}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error('API Error');
        const json = await res.json();
        
        if (json.status === 'success' && isMounted) {
          setData(json);
        }
      } catch (err) {
        console.error(err);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    fetchChart();
    
    return () => { isMounted = false; };
  }, [stockCode, stockName]);

  if (!stockCode) {
    return (
      <div className="h-full flex items-center justify-center text-text-muted border-2 border-dashed border-border-color rounded-2xl bg-panel">
        왼쪽에서 종목에 대해 질문하면 대시보드가 표시됩니다.
      </div>
    );
  }

  if (loading || !data) {
    return (
      <div className="h-full bg-panel backdrop-blur-md border border-border-color rounded-2xl p-6 flex flex-col gap-6 animate-pulse">
        <div className="flex justify-between items-center">
          <div className="h-8 bg-slate-700/50 rounded w-1/3"></div>
          <div className="flex gap-4">
            <div className="h-6 bg-slate-700/50 rounded w-16"></div>
            <div className="h-6 bg-slate-700/50 rounded w-16"></div>
          </div>
        </div>
        <div className="flex-1 bg-slate-700/20 rounded-xl"></div>
        <div className="h-24 bg-slate-700/30 rounded-xl"></div>
      </div>
    );
  }

  const { history, latest } = data;

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-slate-800 border border-slate-600 p-3 rounded shadow-lg text-sm">
          <p className="text-slate-300 mb-2 font-bold">{label}</p>
          {payload.map((entry, idx) => (
            <div key={idx} className="flex gap-4 justify-between" style={{ color: entry.color }}>
              <span>{entry.name}:</span>
              <span className="font-mono">{Number(entry.value).toLocaleString()}</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-full bg-panel backdrop-blur-md rounded-2xl border border-border-color p-6 shadow-2xl flex flex-col">
      <div className="flex justify-between items-end mb-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white">{stockName || stockCode} 기업 분석 대시보드</h2>
        </div>
        <div className="flex gap-6 text-right">
          <div>
            <p className="text-xs text-text-muted mb-1 font-semibold uppercase">현재가</p>
            <p className="text-lg text-white font-mono">{latest?.price?.toLocaleString() || '-'}</p>
          </div>
          <div>
            <p className="text-xs text-text-muted mb-1 font-semibold uppercase">RSI (14)</p>
            <p className={`text-lg font-mono ${latest?.rsi < 35 ? 'text-green-400' : latest?.rsi > 70 ? 'text-red-400' : 'text-white'}`}>
              {latest?.rsi || '-'}
            </p>
          </div>
          <div>
            <p className="text-xs text-text-muted mb-1 font-semibold uppercase">거래량비</p>
            <p className="text-lg text-white font-mono">{latest?.volume_ratio || '-'}x</p>
          </div>
        </div>
      </div>

      <div className="flex-1 bg-slate-900/50 rounded-xl p-4 border border-border-color/50 mb-6 relative min-h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis dataKey="date" stroke="#94a3b8" fontSize={11} tickMargin={8} minTickGap={30} />
            <YAxis domain={['auto', 'auto']} stroke="#94a3b8" fontSize={11} width={60} tickFormatter={(v) => v.toLocaleString()} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
            
            <Line type="monotone" name="종가" dataKey="close" stroke="#38bdf8" strokeWidth={2} dot={false} isAnimationActive={false} />
            {showMA5 && <Line type="monotone" name="MA5" dataKey="ma5" stroke="#f472b6" strokeWidth={1.5} dot={false} isAnimationActive={false} />}
            {showMA20 && <Line type="monotone" name="MA20" dataKey="ma20" stroke="#fbbf24" strokeWidth={1.5} strokeDasharray="5 5" dot={false} isAnimationActive={false} />}
            {showMA60 && <Line type="monotone" name="MA60" dataKey="ma60" stroke="#a78bfa" strokeWidth={1.5} dot={false} isAnimationActive={false} />}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex-shrink-0 grid grid-cols-2 gap-4">
        {/* Toggle Switches */}
        <div className="bg-slate-800/40 rounded-xl p-4 border border-border-color flex flex-col gap-3 justify-center">
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-sm font-medium text-slate-300">5일 이동평균선</span>
            <input type="checkbox" className="sr-only peer" checked={showMA5} onChange={() => setShowMA5(!showMA5)} />
            <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-pink-400 relative"></div>
          </label>
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-sm font-medium text-slate-300">20일 이동평균선</span>
            <input type="checkbox" className="sr-only peer" checked={showMA20} onChange={() => setShowMA20(!showMA20)} />
            <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-400 relative"></div>
          </label>
          <label className="flex items-center justify-between cursor-pointer">
            <span className="text-sm font-medium text-slate-300">60일 이동평균선</span>
            <input type="checkbox" className="sr-only peer" checked={showMA60} onChange={() => setShowMA60(!showMA60)} />
            <div className="w-11 h-6 bg-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-400 relative"></div>
          </label>
        </div>
        
        {/* Summary Card */}
        <div className="bg-slate-800/40 rounded-xl p-4 border border-border-color">
           <h3 className="text-sm font-bold text-white mb-2 pb-1 border-b border-slate-700">시그널 요약</h3>
           <ul className="text-sm space-y-2 text-slate-300">
             <li className="flex justify-between"><span>현재가:</span> <span className="font-mono text-white">{latest?.price?.toLocaleString() || '-'}</span></li>
             <li className="flex justify-between"><span>추세 (MA20):</span> <span className={latest?.price > latest?.ma20 ? 'text-red-400 font-bold' : 'text-blue-400 font-bold'}>{latest?.price > latest?.ma20 ? '상승추세 (정배열)' : '조정구간 (역배열)'}</span></li>
             <li className="flex justify-between"><span>시장심리 (RSI):</span> <span className="font-mono">{latest?.rsi > 70 ? '단기 고점 (과매수)' : latest?.rsi < 30 ? '반등 대기 (과매도)' : '박스권 횡보'}</span></li>
             <li className="flex justify-between"><span>수급 (거래량):</span> <span className="font-mono">{latest?.volume_ratio}배 (평소 대비)</span></li>
           </ul>
        </div>
      </div>
    </div>
  );
}

function App() {
  return (
    <div className="flex h-screen p-4 lg:p-6 gap-6 max-w-[1920px] mx-auto w-full overflow-hidden">
      <div className="w-full lg:w-[450px] xl:w-[500px] flex-shrink-0 h-full">
        <ChatPanel />
      </div>
      <div className="hidden lg:block flex-1 min-w-0 h-full">
        <DashboardPanel />
      </div>
    </div>
  );
}

export default App;
