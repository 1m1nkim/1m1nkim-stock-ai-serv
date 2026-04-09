import { create } from 'zustand';

export const useStore = create((set) => ({
  stockCode: null,
  stockName: null,
  latestData: null,
  setStockInfo: (code, name, latest) => set({ 
    stockCode: code, 
    stockName: name, 
    latestData: latest 
  }),
}));
