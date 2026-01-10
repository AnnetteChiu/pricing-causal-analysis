"""
進階定價策略因果推論方法
Advanced Causal Inference Methods for Pricing Strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class AdvancedPricingAnalysis:
    """進階定價分析方法"""
    
    def __init__(self, data):
        self.data = data
        self.results = {}
    
    def difference_in_differences(self, pre_period_end='2023-06-30', treatment_start='2023-07-01'):
        """差分差分法分析 (Difference-in-Differences)"""
        print("=== 差分差分法分析 ===")
        
        # 生成時間序列數據
        np.random.seed(42)
        n_periods = 12  # 12個月
        n_stores = 100  # 100家店鋪
        
        # 創建面板數據
        dates = pd.date_range('2023-01-01', periods=n_periods, freq='M')
        stores = range(n_stores)
        
        panel_data = []
        for store in stores:
            for date in dates:
                # 隨機分配處理組（50%的店鋪）
                treated = store < n_stores // 2
                post_treatment = date >= pd.to_datetime(treatment_start)
                
                # 基礎銷量
                base_sales = 1000 + np.random.normal(0, 100)
                
                # 店鋪固定效應
                if not hasattr(self, '_store_effects'):
                    self._store_effects = {}
                if store not in self._store_effects:
                    self._store_effects[store] = np.random.normal(0, 50)
                store_effect = self._store_effects[store]
                
                # 時間趨勢
                time_trend = (date - dates[0]).days * 0.1
                
                # 處理效應 (DID效應)
                treatment_effect = 150 if (treated and post_treatment) else 0
                
                # 最終銷量
                sales = base_sales + store_effect + time_trend + treatment_effect + np.random.normal(0, 30)
                
                panel_data.append({
                    'store_id': store,
                    'date': date,
                    'treated': treated,
                    'post': post_treatment,
                    'sales': max(sales, 0),
                    'price': 95 if (treated and post_treatment) else 100
                })
        
        panel_df = pd.DataFrame(panel_data)
        
        # DID回歸
        panel_df['treated_post'] = panel_df['treated'] * panel_df['post']
        
        # 準備回歸變數
        X = panel_df[['treated', 'post', 'treated_post']].astype(int)
        y = panel_df['sales']
        
        model = LinearRegression()
        model.fit(X, y)
        
        did_effect = model.coef_[2]  # treated_post係數
        
        print(f"DID估計的處理效應: {did_effect:.2f}")
        print(f"模型R²: {model.score(X, y):.4f}")
        
        # 平行趨勢檢驗（簡化版）
        pre_treatment = panel_df[panel_df['post'] == False]
        treated_trend = pre_treatment[pre_treatment['treated'] == True].groupby('date')['sales'].mean()
        control_trend = pre_treatment[pre_treatment['treated'] == False].groupby('date')['sales'].mean()
        
        # 計算趨勢差異
        trend_diff = (treated_trend.iloc[-1] - treated_trend.iloc[0]) - (control_trend.iloc[-1] - control_trend.iloc[0])
        
        print(f"處理前趨勢差異: {trend_diff:.2f} (接近0表示滿足平行趨勢假設)")
        
        self.results['did_analysis'] = {
            'did_effect': did_effect,
            'trend_difference': trend_diff,
            'panel_data': panel_df
        }
        
        return panel_df
    
    def instrumental_variables(self):
        """工具變數法分析"""
        print("\n=== 工具變數法分析 ===")
        
        # 創建工具變數（例如：供應商成本衝擊）
        np.random.seed(42)
        n = len(self.data)
        
        # 工具變數：外生的成本衝擊
        cost_shock = np.random.normal(0, 10, n)
        
        # 第一階段：工具變數對內生變數（價格）的回歸
        # 價格 = f(工具變數, 其他控制變數)
        price_endogenous = (self.data['price'] + 
                           0.5 * cost_shock +  # 工具變數影響
                           0.3 * self.data['competitor_price'] +
                           np.random.normal(0, 5, n))
        
        # 第一階段回歸
        X_first = np.column_stack([cost_shock, self.data['competitor_price']])
        first_stage = LinearRegression()
        first_stage.fit(X_first, price_endogenous)
        
        # 預測價格（去除內生性）
        predicted_price = first_stage.predict(X_first)
        
        # 第二階段：結果變數對預測的內生變數回歸
        X_second = np.column_stack([predicted_price, self.data['competitor_price']])
        second_stage = LinearRegression()
        second_stage.fit(X_second, self.data['sales_volume'])
        
        iv_price_effect = second_stage.coef_[0]
        
        # F統計量檢驗工具變數強度
        f_stat = np.var(predicted_price) / np.var(price_endogenous - predicted_price) * (n - 2)
        
        print(f"IV估計的價格效應: {iv_price_effect:.4f}")
        print(f"第一階段F統計量: {f_stat:.2f} (>10表示強工具變數)")
        print(f"第二階段R²: {second_stage.score(X_second, self.data['sales_volume']):.4f}")
        
        self.results['iv_analysis'] = {
            'iv_price_effect': iv_price_effect,
            'f_statistic': f_stat,
            'first_stage_r2': first_stage.score(X_first, price_endogenous)
        }
    
    def regression_discontinuity(self):
        """回歸不連續設計"""
        print("\n=== 回歸不連續設計分析 ===")
        
        # 創建跑變數（例如：客戶忠誠度分數）
        np.random.seed(42)
        n = len(self.data)
        
        # 跑變數：客戶忠誠度分數 (0-100)
        loyalty_score = np.random.uniform(0, 100, n)
        
        # 閾值：分數>=70的客戶獲得折扣
        threshold = 70
        discount_eligible = loyalty_score >= threshold
        
        # 處理效應：符合條件的客戶獲得10%折扣
        discounted_price = np.where(discount_eligible, 
                                   self.data['price'] * 0.9, 
                                   self.data['price'])
        
        # 生成結果變數（銷量）
        # 包含跑變數的連續效應和處理的跳躍效應
        base_effect = 1000 + 2 * loyalty_score  # 連續效應
        treatment_effect = 200 * discount_eligible  # 跳躍效應
        sales_rd = base_effect + treatment_effect + np.random.normal(0, 50, n)
        
        # RD估計：在閾值附近的局部線性回歸
        bandwidth = 10  # 帶寬
        near_threshold = np.abs(loyalty_score - threshold) <= bandwidth
        
        if np.sum(near_threshold) > 10:  # 確保有足夠樣本
            rd_data = pd.DataFrame({
                'loyalty_score': loyalty_score[near_threshold],
                'treated': discount_eligible[near_threshold],
                'sales': sales_rd[near_threshold],
                'running_var': loyalty_score[near_threshold] - threshold
            })
            
            # 局部線性回歸
            X_rd = np.column_stack([
                rd_data['running_var'],
                rd_data['treated'],
                rd_data['running_var'] * rd_data['treated']  # 交互項
            ])
            
            rd_model = LinearRegression()
            rd_model.fit(X_rd, rd_data['sales'])
            
            rd_effect = rd_model.coef_[1]  # 處理效應
            
            print(f"RD估計的處理效應: {rd_effect:.2f}")
            print(f"使用帶寬: {bandwidth}")
            print(f"閾值附近樣本數: {len(rd_data)}")
            
            self.results['rd_analysis'] = {
                'rd_effect': rd_effect,
                'bandwidth': bandwidth,
                'n_near_threshold': len(rd_data),
                'rd_data': rd_data
            }
        else:
            print("閾值附近樣本數不足，無法進行RD分析")
    
    def synthetic_control(self):
        """合成控制法"""
        print("\n=== 合成控制法分析 ===")
        
        # 創建多個市場的時間序列數據
        np.random.seed(42)
        n_markets = 10
        n_periods = 24  # 24個月
        treatment_period = 12  # 第12個月開始處理
        
        # 生成各市場的銷售數據
        markets_data = {}
        
        for market in range(n_markets):
            sales_series = []
            for period in range(n_periods):
                # 基礎趨勢
                base_trend = 1000 + period * 10
                
                # 市場特定效應
                market_effect = np.random.normal(0, 100)
                
                # 處理效應（只有市場0在第12期後受到處理）
                if market == 0 and period >= treatment_period:
                    treatment_effect = 200
                else:
                    treatment_effect = 0
                
                # 隨機誤差
                noise = np.random.normal(0, 50)
                
                sales = base_trend + market_effect + treatment_effect + noise
                sales_series.append(max(sales, 0))
            
            markets_data[f'market_{market}'] = sales_series
        
        markets_df = pd.DataFrame(markets_data)
        
        # 合成控制：使用處理前期間的數據來構建合成控制組
        pre_treatment_data = markets_df.iloc[:treatment_period]
        treated_pre = pre_treatment_data['market_0'].values
        donors_pre = pre_treatment_data.drop('market_0', axis=1).values
        
        # 簡化的合成控制權重計算（最小化處理前期間的差異）
        from scipy.optimize import minimize
        
        def objective(weights):
            synthetic = donors_pre @ weights
            return np.sum((treated_pre - synthetic) ** 2)
        
        # 約束條件：權重和為1，且非負
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(donors_pre.shape[1])]
        
        result = minimize(objective, 
                         x0=np.ones(donors_pre.shape[1]) / donors_pre.shape[1],
                         bounds=bounds, 
                         constraints=constraints)
        
        optimal_weights = result.x
        
        # 構建合成控制組
        donors_all = markets_df.drop('market_0', axis=1).values
        synthetic_control = donors_all @ optimal_weights
        
        # 計算處理效應
        post_treatment_effect = (markets_df['market_0'].iloc[treatment_period:].values - 
                               synthetic_control[treatment_period:]).mean()
        
        print(f"合成控制法估計的平均處理效應: {post_treatment_effect:.2f}")
        print(f"處理前期間RMSPE: {np.sqrt(objective(optimal_weights)):.2f}")
        
        # 顯示權重
        weight_df = pd.DataFrame({
            'market': [f'market_{i+1}' for i in range(len(optimal_weights))],
            'weight': optimal_weights
        }).sort_values('weight', ascending=False)
        
        print("\n合成控制權重 (前5名):")
        print(weight_df.head())
        
        self.results['synthetic_control'] = {
            'treatment_effect': post_treatment_effect,
            'weights': optimal_weights,
            'markets_data': markets_df,
            'synthetic_series': synthetic_control
        }
    
    def machine_learning_causal(self):
        """機器學習因果推論方法"""
        print("\n=== 機器學習因果推論 ===")
        
        # Double Machine Learning (DML)
        from sklearn.model_selection import cross_val_predict
        
        # 準備特徵
        features = ['competitor_price', 'marketing_spend', 'inventory_level']
        season_dummies = pd.get_dummies(self.data['season'], prefix='season')
        segment_dummies = pd.get_dummies(self.data['customer_segment'], prefix='segment')
        
        X = pd.concat([
            self.data[features],
            season_dummies,
            segment_dummies
        ], axis=1)
        
        D = self.data['price_treatment']  # 處理變數
        Y = self.data['sales_volume']     # 結果變數
        
        # 第一步：預測處理變數
        rf_d = RandomForestRegressor(n_estimators=100, random_state=42)
        D_pred = cross_val_predict(rf_d, X, D, cv=5)
        D_residual = D - D_pred
        
        # 第二步：預測結果變數
        rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
        Y_pred = cross_val_predict(rf_y, X, Y, cv=5)
        Y_residual = Y - Y_pred
        
        # 第三步：殘差回歸
        dml_model = LinearRegression()
        dml_model.fit(D_residual.reshape(-1, 1), Y_residual)
        
        dml_effect = dml_model.coef_[0]
        
        print(f"Double ML估計的處理效應: {dml_effect:.2f}")
        
        # Causal Forest (簡化版)
        # 估計異質性處理效應
        heterogeneous_effects = []
        
        for segment in self.data['customer_segment'].unique():
            segment_mask = self.data['customer_segment'] == segment
            segment_treated = self.data[segment_mask & (self.data['price_treatment'] == 1)]['sales_volume']
            segment_control = self.data[segment_mask & (self.data['price_treatment'] == 0)]['sales_volume']
            
            if len(segment_treated) > 0 and len(segment_control) > 0:
                segment_effect = segment_treated.mean() - segment_control.mean()
                heterogeneous_effects.append((segment, segment_effect))
        
        print(f"\n異質性處理效應:")
        for segment, effect in heterogeneous_effects:
            print(f"- {segment}: {effect:.2f}")
        
        self.results['ml_causal'] = {
            'dml_effect': dml_effect,
            'heterogeneous_effects': dict(heterogeneous_effects)
        }
    
    def visualize_advanced_results(self):
        """可視化進階分析結果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('進階因果推論方法結果', fontsize=16)
        
        # 1. DID分析結果
        if 'did_analysis' in self.results:
            panel_data = self.results['did_analysis']['panel_data']
            
            # 計算各組平均值
            group_means = panel_data.groupby(['treated', 'post'])['sales'].mean().unstack()
            
            x = [0, 1]  # 處理前後
            axes[0, 0].plot(x, [group_means.loc[False, False], group_means.loc[False, True]], 
                           'o-', label='控制組', linewidth=2)
            axes[0, 0].plot(x, [group_means.loc[True, False], group_means.loc[True, True]], 
                           'o-', label='處理組', linewidth=2)
            axes[0, 0].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='處理開始')
            axes[0, 0].set_title('差分差分法 (DID)')
            axes[0, 0].set_xlabel('時期')
            axes[0, 0].set_ylabel('平均銷量')
            axes[0, 0].set_xticks([0, 1])
            axes[0, 0].set_xticklabels(['處理前', '處理後'])
            axes[0, 0].legend()
        
        # 2. 回歸不連續設計
        if 'rd_analysis' in self.results:
            rd_data = self.results['rd_analysis']['rd_data']
            
            # 分別繪製處理組和控制組
            treated_data = rd_data[rd_data['treated'] == True]
            control_data = rd_data[rd_data['treated'] == False]
            
            axes[0, 1].scatter(control_data['loyalty_score'], control_data['sales'], 
                              alpha=0.6, label='控制組', s=20)
            axes[0, 1].scatter(treated_data['loyalty_score'], treated_data['sales'], 
                              alpha=0.6, label='處理組', s=20)
            axes[0, 1].axvline(x=70, color='red', linestyle='--', label='閾值')
            axes[0, 1].set_title('回歸不連續設計 (RD)')
            axes[0, 1].set_xlabel('忠誠度分數')
            axes[0, 1].set_ylabel('銷量')
            axes[0, 1].legend()
        
        # 3. 合成控制法
        if 'synthetic_control' in self.results:
            markets_data = self.results['synthetic_control']['markets_data']
            synthetic_series = self.results['synthetic_control']['synthetic_series']
            
            periods = range(len(markets_data))
            axes[1, 0].plot(periods, markets_data['market_0'], 'o-', label='處理組', linewidth=2)
            axes[1, 0].plot(periods, synthetic_series, 's-', label='合成控制組', linewidth=2)
            axes[1, 0].axvline(x=12, color='red', linestyle='--', alpha=0.7, label='處理開始')
            axes[1, 0].set_title('合成控制法')
            axes[1, 0].set_xlabel('時期')
            axes[1, 0].set_ylabel('銷量')
            axes[1, 0].legend()
        
        # 4. 異質性處理效應
        if 'ml_causal' in self.results:
            het_effects = self.results['ml_causal']['heterogeneous_effects']
            segments = list(het_effects.keys())
            effects = list(het_effects.values())
            
            bars = axes[1, 1].bar(segments, effects, color=['skyblue', 'lightcoral', 'lightgreen'])
            axes[1, 1].set_title('異質性處理效應')
            axes[1, 1].set_xlabel('客戶群體')
            axes[1, 1].set_ylabel('處理效應')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # 添加數值標籤
            for bar, effect in zip(bars, effects):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                               f'{effect:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def generate_advanced_report(self):
        """生成進階分析報告"""
        print("\n" + "="*60)
        print("進階定價策略因果推論分析報告")
        print("="*60)
        
        print("\n方法比較:")
        methods_effects = {}
        
        if 'did_analysis' in self.results:
            methods_effects['差分差分法'] = self.results['did_analysis']['did_effect']
            
        if 'iv_analysis' in self.results:
            methods_effects['工具變數法'] = self.results['iv_analysis']['iv_price_effect'] * (-10)  # 轉換為處理效應
            
        if 'rd_analysis' in self.results:
            methods_effects['回歸不連續'] = self.results['rd_analysis']['rd_effect']
            
        if 'synthetic_control' in self.results:
            methods_effects['合成控制法'] = self.results['synthetic_control']['treatment_effect']
            
        if 'ml_causal' in self.results:
            methods_effects['Double ML'] = self.results['ml_causal']['dml_effect']
        
        for method, effect in methods_effects.items():
            print(f"- {method}: {effect:.2f}")
        
        print(f"\n方法學建議:")
        print("- 差分差分法適用於有時間變異和橫截面變異的面板數據")
        print("- 工具變數法可以處理價格內生性問題")
        print("- 回歸不連續設計適用於有明確閾值規則的情況")
        print("- 合成控制法適用於少數處理單位的情況")
        print("- 機器學習方法可以捕捉複雜的非線性關係和異質性效應")
        
        if 'ml_causal' in self.results:
            print(f"\n異質性分析:")
            het_effects = self.results['ml_causal']['heterogeneous_effects']
            for segment, effect in het_effects.items():
                print(f"- {segment}客戶群體的處理效應: {effect:.2f}")

def run_advanced_analysis(basic_analyzer):
    """運行進階分析"""
    print("開始進階定價策略因果推論分析...")
    
    # 創建進階分析實例
    advanced_analyzer = AdvancedPricingAnalysis(basic_analyzer.data)
    
    # 執行各種進階分析
    advanced_analyzer.difference_in_differences()
    advanced_analyzer.instrumental_variables()
    advanced_analyzer.regression_discontinuity()
    advanced_analyzer.synthetic_control()
    advanced_analyzer.machine_learning_causal()
    
    # 可視化結果
    advanced_analyzer.visualize_advanced_results()
    
    # 生成報告
    advanced_analyzer.generate_advanced_report()
    
    return advanced_analyzer

if __name__ == "__main__":
    # 需要先運行基礎分析
    print("請先運行 pricing_causal_analysis.py 獲取基礎數據")