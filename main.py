# AGENTIC AI SYSTEM WITH RAG - UNIVERSAL CONTEXT-BASED QUESTION HANDLER
# NOW ANSWERS ANY TYPE OF QUESTION WITH RAG ENHANCEMENT
# Smart question routing + RAG for ALL questions


import json
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Any, Dict, List, Union
import warnings
warnings.filterwarnings('ignore')


try:
    import chromadb
    from chromadb.config import Settings
    chromadb_available = True
except:
    chromadb_available = False


try:
    from sklearn.linear_model import LinearRegression
    from statsmodels.tsa.arima.model import ARIMA
    statsmodels_available = True
except:
    statsmodels_available = False


# ======================= STYLING - FIXED FOR VISIBLE TEXT =======================


def apply_custom_styling():
    """Apply CSS with VISIBLE dark text on light background"""
    st.markdown("""
    <style>
        h1 { font-size: 2.5rem !important; color: #1f77b4 !important; font-weight: 700 !important; }
        h2 { font-size: 1.8rem !important; color: #2c3e50 !important; border-bottom: 2px solid #1f77b4; padding-bottom: 0.5rem !important; }
        h3 { font-size: 1.3rem !important; color: #34495e !important; font-weight: 600 !important; }
        
        .result-box { 
            background-color: #ffffff !important; 
            border: 2px solid #1f77b4 !important; 
            padding: 1.5rem !important; 
            border-radius: 8px !important; 
            margin: 1rem 0 !important;
            color: #000000 !important;
        }
        
        .forecast-box { 
            background-color: #f0fff4 !important; 
            border: 2px solid #00a86b !important; 
            padding: 1.5rem !important; 
            border-radius: 8px !important; 
            margin: 1rem 0 !important;
            color: #000000 !important;
        }
        
        .rag-box { 
            background-color: #fff3e0 !important; 
            border: 2px solid #ff9800 !important; 
            padding: 1.5rem !important; 
            border-radius: 8px !important; 
            margin: 1rem 0 !important;
            color: #000000 !important;
        }
        
        .error-box { 
            background-color: #fff5f5 !important; 
            border: 2px solid #ff6b6b !important; 
            padding: 1.5rem !important; 
            border-radius: 8px !important; 
            margin: 1rem 0 !important;
            color: #000000 !important;
        }
        
        .code-text { 
            font-family: 'Courier New', monospace; 
            font-size: 15px; 
            line-height: 1.9; 
            color: #000000 !important;
            white-space: pre-wrap; 
            word-wrap: break-word; 
            font-weight: 500 !important;
            background-color: transparent !important;
        }
        
        pre {
            color: #000000 !important;
            background-color: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)


# ======================= RAG AGENT =======================


class RAGAgent:
    """Retrieval Augmented Generation Agent"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        
        if chromadb_available:
            try:
                settings = Settings(
                    chroma_db_impl="duckdb",
                    persist_directory="./chroma_db",
                    anonymized_telemetry=False
                )
                self.client = chromadb.Client(settings)
            except:
                pass
    
    def index_documents(self, documents: List[str], doc_ids: List[str] = None) -> bool:
        """Index documents into ChromaDB"""
        if not chromadb_available or not self.client:
            return False
        
        try:
            self.collection = self.client.get_or_create_collection(
                name="business_docs",
                metadata={"hnsw:space": "cosine"}
            )
            
            if doc_ids is None:
                doc_ids = [f"doc_{i}" for i in range(len(documents))]
            
            self.collection.add(documents=documents, ids=doc_ids)
            return True
        except:
            return False
    
    def retrieve(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant documents"""
        if not chromadb_available or not self.client or not self.collection:
            return []
        
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            if results and results['documents']:
                return results['documents'][0]
            return []
        except:
            return []
    
    def generate_answer_with_context(self, question: str, retrieved_docs: List[str], data_context: str = "") -> str:
        """Generate answer using retrieved context"""
        if not retrieved_docs:
            return ""
        
        context = "\n".join([f"• {doc}" for doc in retrieved_docs])
        
        answer = f"📚 RAG-ENHANCED ANSWER\n"
        answer += "=" * 60 + "\n\n"
        answer += f"Question: {question}\n\n"
        answer += f"📖 Retrieved Business Context:\n{context}\n"
        
        if data_context:
            answer += f"\n📊 Data Context:\n{data_context}\n"
        
        answer += f"\n💡 SYNTHESIS:\n"
        answer += "Based on business documents and data analysis, here's the strategic insight:\n"
        
        return answer


# ======================= COLUMN FINDER =======================


class ColumnFinder:
    """Auto-detect columns"""
    
    @staticmethod
    def find_column(df: pd.DataFrame, keywords: List[str]) -> str:
        for col in df.columns:
            for keyword in keywords:
                if keyword.lower() in col.lower():
                    return col
        return None
    
    @staticmethod
    def find_metric_column(df: pd.DataFrame, metric_type: str) -> str:
        metrics = {
            "sales": ["sales", "revenue", "amount", "total"],
            "profit": ["profit", "margin"],
            "rating": ["rating", "score", "satisfaction"],
            "quantity": ["units", "quantity", "sold"]
        }
        keywords = metrics.get(metric_type, [metric_type])
        return ColumnFinder.find_column(df, keywords)
    
    @staticmethod
    def find_time_column(df: pd.DataFrame) -> tuple:
        for col in df.columns:
            if "quarter" in col.lower() or col.lower() in ["q1", "q2", "q3", "q4"]:
                return col, "quarter"
        for col in df.columns:
            if "month" in col.lower() or "date" in col.lower():
                return col, "month"
        return None, None


# ======================= AGENTS =======================


class ETLAgent:
    def analyze(self, df: pd.DataFrame) -> Dict:
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "status": "✅ CLEAN" if df.isnull().sum().sum() == 0 else "⚠️ HAS ISSUES"
        }


class AnalysisAgent:
    """UNIVERSAL ANALYSIS AGENT - Handles ANY question type"""
    
    def analyze(self, df: pd.DataFrame, question: str) -> tuple:
        """
        Returns (answer, data_context) tuple
        Now handles ANY question type, not just specific keywords
        """
        q = question.lower()
        answer = ""
        data_context = ""
        
        try:
            # ===== WHY QUESTIONS =====
            if "why" in q and ("performing" in q or "best" in q or "leading" in q or "strong" in q):
                answer, data_context = self._analyze_why_performing(df)
            
            # ===== AVERAGE QUESTIONS =====
            elif "average" in q and "rating" in q and "category" in q:
                answer, data_context = self._analyze_average_rating(df)
            
            # ===== HIGHEST/MAXIMUM QUESTIONS =====
            elif ("highest" in q or "maximum" in q) and "category" in q:
                answer, data_context = self._analyze_highest_sales(df)
            
            # ===== FORECAST QUESTIONS =====
            elif "forecast" in q or "predict" in q or "future" in q:
                # This will be handled by ForecastAgent, return empty
                answer = ""
                data_context = ""
            
            # ===== PRODUCT STRENGTHS (NEW!) =====
            elif "strength" in q or "strong" in q or "advantage" in q:
                answer, data_context = self._analyze_product_strengths(df)
            
            # ===== COMPARISON QUESTIONS (NEW!) =====
            elif "compare" in q or "vs" in q or "versus" in q:
                answer, data_context = self._analyze_comparison(df)
            
            # ===== REVENUE/SALES QUESTIONS (NEW!) =====
            elif ("revenue" in q or "sales" in q or "income" in q) and ("total" in q or "by" in q):
                answer, data_context = self._analyze_revenue(df)
            
            # ===== PERFORMANCE QUESTIONS (NEW!) =====
            elif "performance" in q or "perform" in q or "rating" in q:
                answer, data_context = self._analyze_performance(df)
            
            # ===== CATCH-ALL: UNIVERSAL QUESTION HANDLER (NEW!) =====
            else:
                answer, data_context = self._analyze_generic_question(df, question)
        
        except Exception as e:
            answer = f"❌ Analysis Error: {str(e)}"
            data_context = ""
        
        if not answer:
            answer = "💡 Try questions like:\n• 'Why is Electronics performing best?'\n• 'Show average rating by category'\n• 'What are our product strengths?'\n• 'Compare Electronics vs Furniture'\n• 'Forecast sales next 3 quarters'"
        
        return answer, data_context
    
    def _analyze_why_performing(self, df: pd.DataFrame) -> tuple:
        """Analyze why category is performing"""
        try:
            cat_col = ColumnFinder.find_column(df, ["category", "type", "product"])
            rating_col = ColumnFinder.find_column(df, ["rating", "score"])
            sales_col = ColumnFinder.find_metric_column(df, "sales")
            
            if cat_col and (rating_col or sales_col):
                if rating_col:
                    best_cat = df.groupby(cat_col)[rating_col].mean().idxmax()
                    best_rating = df.groupby(cat_col)[rating_col].mean().max()
                
                if sales_col:
                    best_sales = df.groupby(cat_col)[sales_col].sum().max()
                
                answer = f"💡 WHY IS {best_cat.upper()} PERFORMING BEST?\n"
                answer += "=" * 60 + "\n\n"
                answer += "📊 KEY PERFORMANCE METRICS:\n"
                
                if rating_col:
                    answer += f"• Customer Rating: {best_rating:.2f}/5.0 ⭐\n"
                if sales_col:
                    answer += f"• Total Sales: ${best_sales:,.0f}\n"
                
                answer += "\n📈 COMPARATIVE ANALYSIS:\n"
                
                if rating_col and sales_col:
                    all_ratings = df.groupby(cat_col)[rating_col].mean().sort_values(ascending=False)
                    all_sales = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False)
                    
                    for cat in all_ratings.index[:3]:
                        rating = all_ratings[cat]
                        sales = all_sales.get(cat, 0) if cat in all_sales.index else 0
                        answer += f"• {cat}: Rating {rating:.2f}/5.0 | Sales ${sales:,.0f}\n"
                
                answer += "\n💡 KEY REASONS FOR SUCCESS:\n"
                answer += f"• Highest customer satisfaction ({best_rating:.2f}/5.0)\n"
                answer += f"• Strong revenue generation (${best_sales:,.0f})\n"
                answer += f"• Market leadership in category\n"
                answer += f"• Consistent customer preference\n"
                
                data_context = f"Best Category: {best_cat}\nRating: {best_rating:.2f}\nSales: ${best_sales:,.0f}"
                return answer, data_context
        except:
            pass
        
        return "", ""
    
    def _analyze_average_rating(self, df: pd.DataFrame) -> tuple:
        """Analyze average rating by category"""
        try:
            cat_col = ColumnFinder.find_column(df, ["category", "type"])
            rating_col = ColumnFinder.find_column(df, ["rating", "score"])
            
            if cat_col and rating_col:
                result = df.groupby(cat_col)[rating_col].mean().sort_values(ascending=False)
                answer = "⭐ AVERAGE RATING BY CATEGORY\n"
                answer += "=" * 60 + "\n\n"
                for category, rating in result.items():
                    answer += f"{category}: {rating:.2f}/5.0 ⭐\n"
                
                data_context = "\n".join([f"{cat}: {rating:.2f}" for cat, rating in result.items()])
                return answer, data_context
        except:
            pass
        
        return "", ""
    
    def _analyze_highest_sales(self, df: pd.DataFrame) -> tuple:
        """Analyze highest sales by category"""
        try:
            cat_col = ColumnFinder.find_column(df, ["category", "type"])
            sales_col = ColumnFinder.find_metric_column(df, "sales")
            
            if cat_col and sales_col:
                result = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False)
                answer = "📊 HIGHEST SALES BY CATEGORY\n"
                answer += "=" * 60 + "\n\n"
                for i, (cat, val) in enumerate(result.items(), 1):
                    marker = "🏆" if i == 1 else f"{i}."
                    answer += f"{marker} {cat}: ${val:,.0f}\n"
                
                data_context = "\n".join([f"{cat}: ${val:,.0f}" for cat, val in result.items()])
                return answer, data_context
        except:
            pass
        
        return "", ""
    
    def _analyze_product_strengths(self, df: pd.DataFrame) -> tuple:
        """Analyze product strengths (NEW!)"""
        try:
            cat_col = ColumnFinder.find_column(df, ["category", "type"])
            rating_col = ColumnFinder.find_column(df, ["rating", "score"])
            product_col = ColumnFinder.find_column(df, ["product", "name"])
            
            if cat_col and rating_col:
                answer = "🏆 PRODUCT STRENGTHS ANALYSIS\n"
                answer += "=" * 60 + "\n\n"
                
                # Find top products by rating
                if product_col:
                    top_products = df.nlargest(5, rating_col)[[product_col, cat_col, rating_col]]
                    answer += "⭐ TOP RATED PRODUCTS:\n"
                    for _, row in top_products.iterrows():
                        answer += f"• {row[product_col]} ({row[cat_col]}): {row[rating_col]:.2f}/5.0\n"
                
                # Category strengths
                answer += "\n📊 CATEGORY STRENGTHS:\n"
                cat_ratings = df.groupby(cat_col)[rating_col].mean().sort_values(ascending=False)
                for cat, rating in cat_ratings.items():
                    answer += f"• {cat}: {rating:.2f}/5.0 average rating\n"
                
                data_context = "Analyzed product quality and category performance"
                return answer, data_context
        except:
            pass
        
        return "", ""
    
    def _analyze_comparison(self, df: pd.DataFrame) -> tuple:
        """Analyze comparisons (NEW!)"""
        try:
            cat_col = ColumnFinder.find_column(df, ["category", "type"])
            rating_col = ColumnFinder.find_column(df, ["rating", "score"])
            sales_col = ColumnFinder.find_metric_column(df, "sales")
            
            if cat_col:
                answer = "📊 CATEGORY COMPARISON\n"
                answer += "=" * 60 + "\n\n"
                
                if rating_col and sales_col:
                    comparison = df.groupby(cat_col).agg({
                        rating_col: 'mean',
                        sales_col: 'sum'
                    }).sort_values(by=sales_col, ascending=False)
                    
                    for cat in comparison.index:
                        avg_rating = comparison.loc[cat, rating_col]
                        total_sales = comparison.loc[cat, sales_col]
                        answer += f"• {cat}:\n"
                        answer += f"  - Avg Rating: {avg_rating:.2f}/5.0 ⭐\n"
                        answer += f"  - Total Sales: ${total_sales:,.0f}\n"
                
                data_context = "Category comparison completed"
                return answer, data_context
        except:
            pass
        
        return "", ""
    
    def _analyze_revenue(self, df: pd.DataFrame) -> tuple:
        """Analyze revenue/sales (NEW!)"""
        try:
            cat_col = ColumnFinder.find_column(df, ["category", "type"])
            sales_col = ColumnFinder.find_metric_column(df, "sales")
            
            if cat_col and sales_col:
                answer = "💰 REVENUE ANALYSIS\n"
                answer += "=" * 60 + "\n\n"
                
                by_cat = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False)
                total = by_cat.sum()
                
                answer += "💵 Total Revenue by Category:\n"
                for cat, revenue in by_cat.items():
                    pct = (revenue / total * 100)
                    answer += f"• {cat}: ${revenue:,.0f} ({pct:.1f}%)\n"
                
                answer += f"\n📊 TOTAL REVENUE: ${total:,.0f}\n"
                
                data_context = f"Total revenue: ${total:,.0f}"
                return answer, data_context
        except:
            pass
        
        return "", ""
    
    def _analyze_performance(self, df: pd.DataFrame) -> tuple:
        """Analyze performance metrics (NEW!)"""
        try:
            cat_col = ColumnFinder.find_column(df, ["category", "type"])
            rating_col = ColumnFinder.find_column(df, ["rating", "score"])
            sales_col = ColumnFinder.find_metric_column(df, "sales")
            
            if cat_col:
                answer = "📈 PERFORMANCE METRICS\n"
                answer += "=" * 60 + "\n\n"
                
                if rating_col:
                    avg_ratings = df.groupby(cat_col)[rating_col].mean().sort_values(ascending=False)
                    answer += "⭐ Customer Satisfaction (Rating):\n"
                    for cat, rating in avg_ratings.items():
                        answer += f"• {cat}: {rating:.2f}/5.0\n"
                
                if sales_col:
                    total_sales = df.groupby(cat_col)[sales_col].sum().sort_values(ascending=False)
                    answer += "\n💰 Sales Performance:\n"
                    for cat, sales in total_sales.items():
                        answer += f"• {cat}: ${sales:,.0f}\n"
                
                data_context = "Performance analysis completed"
                return answer, data_context
        except:
            pass
        
        return "", ""
    
    def _analyze_generic_question(self, df: pd.DataFrame, question: str) -> tuple:
        """Handle ANY generic question (FALLBACK!)"""
        try:
            answer = "📊 GENERAL DATA INSIGHTS\n"
            answer += "=" * 60 + "\n\n"
            
            # Get available columns
            cat_col = ColumnFinder.find_column(df, ["category", "type"])
            rating_col = ColumnFinder.find_column(df, ["rating", "score"])
            sales_col = ColumnFinder.find_metric_column(df, "sales")
            
            if cat_col:
                answer += f"📌 Categories Found: {', '.join(df[cat_col].unique())}\n\n"
            
            if rating_col:
                answer += f"⭐ Rating Summary:\n"
                answer += f"• Average: {df[rating_col].mean():.2f}/5.0\n"
                answer += f"• Highest: {df[rating_col].max():.2f}/5.0\n"
                answer += f"• Lowest: {df[rating_col].min():.2f}/5.0\n\n"
            
            if sales_col:
                answer += f"💰 Sales Summary:\n"
                answer += f"• Total: ${df[sales_col].sum():,.0f}\n"
                answer += f"• Average: ${df[sales_col].mean():,.0f}\n"
                answer += f"• Highest: ${df[sales_col].max():,.0f}\n\n"
            
            answer += f"📝 Question: {question}\n"
            answer += "This generic analysis was generated from available data.\n"
            
            data_context = f"Data Summary: {len(df)} records, {len(df.columns)} columns"
            return answer, data_context
        except:
            pass
        
        return "", ""


class ForecastAgent:
    def forecast(self, df: pd.DataFrame, question: str, steps: int = 3) -> str:
        try:
            time_col, time_type = ColumnFinder.find_time_column(df)
            
            if not time_col:
                return ("❌ CANNOT FORECAST\n" + "=" * 60 + "\n\n"
                       "Your data needs Quarter/Month column for forecasting.")
            
            metric_col = ColumnFinder.find_metric_column(df, "sales")
            if not metric_col:
                return "❌ No sales/revenue column found"
            
            ts = df.groupby(time_col)[metric_col].sum().sort_index()
            
            if len(ts) < 2:
                return f"❌ Need at least 2 time periods"
            
            forecast_vals = self._generate_forecast(ts, steps)
            if forecast_vals is None:
                return "❌ Forecasting failed"
            
            answer = f"🔮 FORECAST: {metric_col.upper()} (Next {steps} Periods)\n"
            answer += "=" * 60 + "\n\n"
            
            for i, val in enumerate(forecast_vals, 1):
                answer += f"📈 Period {i}: ${val:,.0f}\n"
            
            # INSIGHT SECTION
            answer += "\n💡 INSIGHT:\n"
            avg_hist = ts.mean()
            avg_fore = np.mean(forecast_vals)
            change = ((avg_fore - avg_hist) / avg_hist) * 100
            
            if change > 0:
                answer += f"📈 Expected growth: +{change:.1f}%\n"
            else:
                answer += f"📉 Expected decline: {change:.1f}%\n"
            
            return answer
        
        except Exception as e:
            return f"❌ FORECAST ERROR: {str(e)}"
    
    @staticmethod
    def _generate_forecast(ts, steps):
        try:
            if statsmodels_available and len(ts) >= 4:
                try:
                    model = ARIMA(ts, order=(1, 1, 1))
                    fitted = model.fit()
                    return fitted.forecast(steps=steps).values
                except:
                    pass
            
            X = np.arange(len(ts)).reshape(-1, 1)
            y = ts.values
            model = LinearRegression()
            model.fit(X, y)
            X_future = np.arange(len(ts), len(ts) + steps).reshape(-1, 1)
            return model.predict(X_future)
        except:
            return None


class RecommendationAgent:
    def generate(self, result_text: str, rag_context: str = "") -> List[str]:
        recs = []
        
        if "BEST" in result_text or "PERFORMING" in result_text:
            recs.append("🚀 Scale: Double down on winning strategy")
            recs.append("📈 Invest: Allocate resources to top performer")
        
        if "FORECAST" in result_text and "growth" in result_text.lower():
            recs.append("📈 Growth expected - plan expansion")
        elif "FORECAST" in result_text and "decline" in result_text.lower():
            recs.append("⚠️ Prepare contingency plans")
        
        if "STRENGTH" in result_text or "RATING" in result_text:
            recs.append("📊 Leverage strengths in marketing")
            recs.append("🎯 Apply best practices across categories")
        
        if "COMPARISON" in result_text:
            recs.append("📊 Focus on top performer strategy")
        
        if rag_context and len(rag_context) > 10:
            recs.append("📚 Review business context for strategy")
        
        if not recs:
            recs.append("✅ Continue monitoring performance")
        
        return recs


class MultiAgentOrchestrator:
    def __init__(self):
        self.etl = ETLAgent()
        self.analysis = AnalysisAgent()
        self.forecast = ForecastAgent()
        self.rag = RAGAgent()
        self.recommend = RecommendationAgent()
    
    def process(self, data, question: str, use_rag: bool = False, rag_docs: List[str] = None) -> Dict:
        df = pd.DataFrame(data)
        etl_result = self.etl.analyze(df)
        
        is_forecast = any(w in question.lower() for w in ["forecast", "predict", "future"])
        
        if is_forecast:
            result_text = self.forecast.forecast(df, question)
            data_context = ""
        else:
            result_text, data_context = self.analysis.analyze(df, question)
        
        rag_result = ""
        if use_rag and rag_docs and chromadb_available and not is_forecast:
            self.rag.index_documents(rag_docs)
            retrieved = self.rag.retrieve(question, n_results=3)
            if retrieved:
                rag_result = self.rag.generate_answer_with_context(question, retrieved, data_context)
        
        final_result = result_text
        if rag_result:
            final_result += "\n\n" + rag_result
        
        recs = self.recommend.generate(result_text, rag_result)
        
        return {
            "etl": etl_result,
            "result": final_result,
            "recommendations": recs,
            "is_forecast": is_forecast,
            "is_error": "❌" in result_text,
            "has_rag": bool(rag_result)
        }


# ======================= STREAMLIT UI =======================


def main():
    st.set_page_config(page_title="🤖 Agentic AI + RAG", layout="wide")
    apply_custom_styling()
    
    st.markdown("# 🤖 Agentic AI System with RAG")
    st.markdown("**5 Autonomous Agents: ETL | Analysis | RAG | Forecast | Recommendations**")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("## 📚 RAG Configuration")
        use_rag = st.checkbox("Enable RAG (Retrieval Augmented Generation)", value=True)
        
        if use_rag:
            st.markdown("### Business Documents")
            rag_docs_text = st.text_area(
                "Paste business intelligence documents (one per line):",
                placeholder="Q1 2024: Strong sales in Electronics...",
                height=200
            )
            rag_docs = [doc.strip() for doc in rag_docs_text.split('\n') if doc.strip()]
        else:
            rag_docs = []
    
    col1, col2 = st.columns([1.5, 1.5])
    
    with col1:
        st.markdown("### 📥 Your Data")
        data_input = st.text_area(
            "JSON Data:",
            placeholder='[{"Category": "Electronics", "Rating": 4.8, ...}, ...]',
            height=220,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### ❓ Your Question")
        question = st.text_area(
            "Question:",
            placeholder="Ask ANY question about your data (e.g., Why is Electronics performing best? What are our product strengths?)",
            height=220,
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    if st.button("🚀 ANALYZE WITH RAG", use_container_width=True, type="primary"):
        if not data_input.strip():
            st.error("❌ Paste your data")
        elif not question.strip():
            st.error("❌ Ask a question")
        else:
            try:
                data = json.loads(data_input)
                orchestrator = MultiAgentOrchestrator()
                result = orchestrator.process(data, question, use_rag=use_rag, rag_docs=rag_docs)
                
                st.markdown("---")
                st.markdown("## 📊 RESULTS")
                
                tabs = st.tabs(["❓ Answer", "🧹 Data Quality", "💡 Recommendations", "📥 JSON"])
                
                with tabs[0]:
                    st.markdown("### Your Question")
                    st.markdown(f"**{question}**")
                    st.markdown("---")
                    st.markdown("### Answer")
                    
                    if result["is_error"]:
                        box_class = "error-box"
                    elif result["has_rag"]:
                        box_class = "rag-box"
                    elif result["is_forecast"]:
                        box_class = "forecast-box"
                    else:
                        box_class = "result-box"
                    
                    st.markdown(
                        f'<div class="{box_class}"><pre class="code-text">{result["result"]}</pre></div>',
                        unsafe_allow_html=True
                    )
                
                with tabs[1]:
                    st.markdown("### Data Quality Report")
                    rep = result["etl"]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("📊 Rows", rep["rows"])
                    col2.metric("📋 Columns", rep["columns"])
                    col3.metric("⚠️ Missing", rep["missing"])
                    col4.metric("🔄 Duplicates", rep["duplicates"])
                    
                    st.markdown("---")
                    st.markdown("**Columns Found:**")
                    for col in rep["column_names"]:
                        st.markdown(f"• `{col}`")
                
                with tabs[2]:
                    st.markdown("### Business Recommendations")
                    for i, rec in enumerate(result["recommendations"], 1):
                        st.markdown(f"**{i}. {rec}**")
                
                with tabs[3]:
                    st.markdown("### Download Results")
                    final = {
                        "timestamp": datetime.now().isoformat(),
                        "question": question,
                        "answer": result["result"],
                        "recommendations": result["recommendations"]
                    }
                    
                    st.download_button(
                        "📥 Download JSON",
                        json.dumps(final, indent=2),
                        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True
                    )
                    st.json(final)
            
            except json.JSONDecodeError:
                st.error("❌ Invalid JSON format")
            except Exception as e:
                st.error(f"❌ System Error: {str(e)}")


if __name__ == "__main__":
    main()