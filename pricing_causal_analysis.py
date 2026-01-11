"""
因果推論在定價策略上的應用 - 數據分析代碼
Causal Inference for Pricing Strategy Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PricingCausalAnalysis:
    """定價策略因果推論分析類"""
    
    def __init__(self, data=None):
        self.data = data
        self.results = {}
        self.is_real_data = data is not None
    
    def load_real_data(self, data_source, **kwargs):
        """
        加載真實數據
        
        Parameters:
        -----------
        data_source : str or pd.DataFrame
            數據源，可以是文件路徑或DataFrame
        **kwargs : dict
            額外參數
        """
        from real_data_loader import RealDataLoader
        
        if isinstance(data_source, pd.DataFrame):
            self.data = data_source
            self.is_real_data = True
            print(f"✅ 已加載DataFrame數據，形狀: {data_source.shape}")
        elif isinstance(data_source, str):
            loader = RealDataLoader()
            
            if data_source.endswith('.csv'):
                self.data = loader.load_csv_data(data_source, **kwargs)
            elif data_source.endswith(('.xlsx', '.xls')):
                self.data = loader.load_excel_data(data_source, **kwargs)
            else:
                raise ValueError("支持的文件格式: .csv, .xlsx, .xls")
            
            self.is_real_data = True
            print(f"✅ 已加載真實數據，形狀: {self.data.shape}")
        else:
            raise ValueError("data_source 必須是文件路徑或DataFrame")
        
        return self.data
    
    def generate_synthetic_data(self, n_samples=10000, seed=42):
        """生成模擬的定價數據"""
        if self.is_real_data:
            print("⚠️  已有真實數據，跳過模擬數據生成")
            return self.data
            
        np.random.seed(seed)
        
        # 基礎特徵
        data = {
            'customer_id': range(n_samples),
            'price': np.random.normal(100, 20, n_samples),
            'competitor_price': np.random.normal(105, 15, n_samples),
            'season': np.random.choice(['春', '夏', '秋', '冬'], n_samples),
            'customer_segment': np.random.choice(['高端', '中端', '低端'], n_samples),
            'marketing_spend': np.random.exponential(1000, n_samples),
            'inventory_level': np.random.uniform(0, 1, n_samples),
        }
        
        # 創建處理變數（價格實驗組）
        treatment_prob = 0.3 + 0.2 * (data['customer_segment'] == '高端').astype(int)
        data['price_treatment'] = np.random.binomial(1, treatment_prob, n_samples)
        
        # 調整實驗組價格
        price_adjustment = np.where(data['price_treatment'] == 1, -10, 0)
        data['price'] = data['price'] + price_adjustment
        
        # 生成銷售量（包含因果效應）
        base_demand = 1000
        price_effect = -2.5 * data['price']
        competitor_effect = 1.5 * data['competitor_price']
        treatment_effect = 150 * data['price_treatment']  # 真實因果效應
        segment_effect = np.where(data['customer_segment'] == '高端', 200,
                                np.where(data['customer_segment'] == '中端', 100, 0))
        marketing_effect = 0.1 * data['marketing_spend']
        seasonal_effect = np.where(data['season'] == '夏', 100,
                                 np.where(data['season'] == '冬', -50, 0))
        
        data['sales_volume'] = (base_demand + price_effect + competitor_effect + 
                               treatment_effect + segment_effect + marketing_effect + 
                               seasonal_effect + np.random.normal(0, 50, n_samples))
        
        # 確保銷售量為正數
        data['sales_volume'] = np.maximum(data['sales_volume'], 0)
        
        # 計算收入和利潤（如果不存在的話）
        if 'revenue' not in self.data.columns:
            self.data['revenue'] = self.data['price'] * self.data['sales_volume']
        
        if 'profit' not in self.data.columns:
            # 假設成本為價格的50%
            self.data['profit'] = self.data['revenue'] - self.data['price'] * 0.5 * self.data['sales_volume']
        
        self.data = pd.DataFrame(data)
        return self.data
    
    def naive_correlation_analysis(self):
        """簡單相關性分析（可能有偏誤）"""
        print("=== 簡單相關性分析 ===")
        
        # 價格與銷量的相關性
        price_sales_corr = self.data['price'].corr(self.data['sales_volume'])
        print(f"價格與銷量相關係數: {price_sales_corr:.4f}")
        
        # 簡單線性回歸
        X = self.data[['price']].values
        y = self.data['sales_volume'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        print(f"簡單回歸係數: {model.coef_[0]:.4f}")
        print(f"R²: {model.score(X, y):.4f}")
        
        self.results['naive_analysis'] = {
            'correlation': price_sales_corr,
            'regression_coef': model.coef_[0],
            'r_squared': model.score(X, y)
        }
    
    def randomized_experiment_analysis(self):
        """隨機實驗分析（A/B測試）"""
        print("\n=== 隨機實驗分析 (A/B測試) ===")
        
        # 確保有必要的列
        if 'revenue' not in self.data.columns:
            self.data['revenue'] = self.data['price'] * self.data['sales_volume']
        
        if 'profit' not in self.data.columns:
            # 假設成本為價格的50%
            self.data['profit'] = self.data['revenue'] - self.data['price'] * 0.5 * self.data['sales_volume']
        
        # 分組統計
        treatment_group = self.data[self.data['price_treatment'] == 1]
        control_group = self.data[self.data['price_treatment'] == 0]
        
        # 平均處理效應 (ATE)
        ate_sales = treatment_group['sales_volume'].mean() - control_group['sales_volume'].mean()
        ate_revenue = treatment_group['revenue'].mean() - control_group['revenue'].mean()
        ate_profit = treatment_group['profit'].mean() - control_group['profit'].mean()
        
        # t檢驗
        t_stat, p_value = stats.ttest_ind(treatment_group['sales_volume'], 
                                         control_group['sales_volume'])
        
        print(f"處理組樣本數: {len(treatment_group)}")
        print(f"對照組樣本數: {len(control_group)}")
        print(f"平均價格差異: {treatment_group['price'].mean() - control_group['price'].mean():.2f}")
        print(f"銷量平均處理效應: {ate_sales:.2f}")
        print(f"收入平均處理效應: {ate_revenue:.2f}")
        print(f"利潤平均處理效應: {ate_profit:.2f}")
        print(f"t統計量: {t_stat:.4f}, p值: {p_value:.4f}")
        
        self.results['experiment_analysis'] = {
            'ate_sales': ate_sales,
            'ate_revenue': ate_revenue,
            'ate_profit': ate_profit,
            't_statistic': t_stat,
            'p_value': p_value
        }
    
    def regression_adjustment_analysis(self):
        """回歸調整分析"""
        print("\n=== 回歸調整分析 ===")
        
        # 準備特徵 - 只使用實際存在的列
        available_features = []
        potential_features = ['competitor_price', 'marketing_spend', 'inventory_level', 
                            'customer_frequency', 'product_avg_price', 'price_relative']
        
        for feature in potential_features:
            if feature in self.data.columns:
                available_features.append(feature)
        
        # 如果沒有額外特徵，至少使用價格
        if not available_features:
            available_features = ['price']
        
        print(f"使用的特徵: {available_features}")
        
        # 創建虛擬變數
        dummy_columns = []
        for col in ['season', 'customer_segment']:
            if col in self.data.columns:
                dummies = pd.get_dummies(self.data[col], prefix=col)
                dummy_columns.append(dummies)
        
        # 組合特徵
        feature_list = [self.data[available_features], self.data[['price_treatment']]]
        feature_list.extend(dummy_columns)
        
        X = pd.concat(feature_list, axis=1)
        y = self.data['sales_volume']
        
        # 線性回歸
        model = LinearRegression()
        model.fit(X, y)
        
        # 獲取處理效應係數
        treatment_coef = model.coef_[X.columns.get_loc('price_treatment')]
        
        print(f"回歸調整後的處理效應: {treatment_coef:.2f}")
        print(f"模型R²: {model.score(X, y):.4f}")
        
        # 特徵重要性
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print("\n特徵重要性 (前5名):")
        print(feature_importance.head())
        
        self.results['regression_analysis'] = {
            'treatment_effect': treatment_coef,
            'r_squared': model.score(X, y),
            'feature_importance': feature_importance
        }
    
    def propensity_score_analysis(self):
        """傾向得分分析"""
        print("\n=== 傾向得分分析 ===")
        
        # 準備特徵用於傾向得分估計 - 只使用實際存在的列
        available_features = []
        potential_features = ['competitor_price', 'marketing_spend', 'inventory_level',
                            'customer_frequency', 'product_avg_price']
        
        for feature in potential_features:
            if feature in self.data.columns:
                available_features.append(feature)
        
        # 如果沒有額外特徵，使用價格
        if not available_features:
            available_features = ['price']
        
        # 創建虛擬變數
        dummy_columns = []
        for col in ['season', 'customer_segment']:
            if col in self.data.columns:
                dummies = pd.get_dummies(self.data[col], prefix=col)
                dummy_columns.append(dummies)
        
        # 組合特徵
        feature_list = [self.data[available_features]]
        feature_list.extend(dummy_columns)
        
        X_ps = pd.concat(feature_list, axis=1)
        
        # 估計傾向得分
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression()
        ps_model.fit(X_ps, self.data['price_treatment'])
        
        propensity_scores = ps_model.predict_proba(X_ps)[:, 1]
        self.data['propensity_score'] = propensity_scores
        
        # 傾向得分匹配 (簡化版)
        treatment_data = self.data[self.data['price_treatment'] == 1].copy()
        control_data = self.data[self.data['price_treatment'] == 0].copy()
        
        matched_pairs = []
        for _, treated in treatment_data.iterrows():
            # 找到最接近的對照組
            distances = np.abs(control_data['propensity_score'] - treated['propensity_score'])
            if len(distances) > 0:
                closest_idx = distances.idxmin()
                matched_pairs.append((treated.name, closest_idx))  # 使用索引而不是customer_id
        
        # 計算匹配後的處理效應
        matched_treatment_sales = []
        matched_control_sales = []
        
        for treated_idx, control_idx in matched_pairs:
            try:
                treated_sales = self.data.loc[treated_idx, 'sales_volume']
                control_sales = self.data.loc[control_idx, 'sales_volume']
                matched_treatment_sales.append(treated_sales)
                matched_control_sales.append(control_sales)
            except (IndexError, KeyError):
                # 如果找不到匹配的記錄，跳過
                continue
        
        ps_ate = np.mean(matched_treatment_sales) - np.mean(matched_control_sales)
        
        print(f"傾向得分匹配後的處理效應: {ps_ate:.2f}")
        print(f"匹配對數: {len(matched_pairs)}")
        
        self.results['propensity_score_analysis'] = {
            'ps_ate': ps_ate,
            'matched_pairs': len(matched_pairs),
            'propensity_scores': propensity_scores
        }
    
    def price_elasticity_analysis(self):
        """價格彈性分析"""
        print("\n=== 價格彈性分析 ===")
        
        # 按客戶群體分析價格彈性
        elasticities = {}
        
        # 檢查是否有客戶分群
        if 'customer_segment' not in self.data.columns:
            # 如果沒有分群，創建一個默認分群
            self.data['customer_segment'] = '全體客戶'
        
        for segment in self.data['customer_segment'].unique():
            segment_data = self.data[self.data['customer_segment'] == segment]
            
            # 確保有足夠的數據點
            if len(segment_data) < 10:
                print(f"   {segment}: 數據點不足，跳過")
                continue
            
            # 過濾掉零值和負值
            valid_data = segment_data[
                (segment_data['price'] > 0) & 
                (segment_data['sales_volume'] > 0)
            ]
            
            if len(valid_data) < 10:
                print(f"   {segment}: 有效數據點不足，跳過")
                continue
            
            try:
                # 計算價格彈性 (% change in quantity / % change in price)
                X = np.log(valid_data['price']).values.reshape(-1, 1)
                y = np.log(valid_data['sales_volume']).values
                
                # 檢查是否有無限值或NaN
                if np.any(np.isinf(X)) or np.any(np.isnan(X)) or np.any(np.isinf(y)) or np.any(np.isnan(y)):
                    print(f"   {segment}: 數據包含無效值，跳過")
                    continue
                
                model = LinearRegression()
                model.fit(X, y)
                
                elasticity = model.coef_[0]
                elasticities[segment] = elasticity
                
                print(f"   {segment}客戶群體價格彈性: {elasticity:.4f}")
                
            except Exception as e:
                print(f"   {segment}: 計算失敗 ({str(e)})，跳過")
                continue
        
        if elasticities:
            self.results['price_elasticity'] = elasticities
        else:
            print("   無法計算價格彈性，可能數據不適合此分析")
    
    def visualize_results(self):
        """結果可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('定價策略因果推論分析結果', fontsize=16)
        
        # 1. 價格分佈對比
        axes[0, 0].hist(self.data[self.data['price_treatment'] == 0]['price'], 
                       alpha=0.7, label='對照組', bins=30)
        axes[0, 0].hist(self.data[self.data['price_treatment'] == 1]['price'], 
                       alpha=0.7, label='實驗組', bins=30)
        axes[0, 0].set_title('價格分佈對比')
        axes[0, 0].set_xlabel('價格')
        axes[0, 0].set_ylabel('頻率')
        axes[0, 0].legend()
        
        # 2. 銷量分佈對比
        axes[0, 1].hist(self.data[self.data['price_treatment'] == 0]['sales_volume'], 
                       alpha=0.7, label='對照組', bins=30)
        axes[0, 1].hist(self.data[self.data['price_treatment'] == 1]['sales_volume'], 
                       alpha=0.7, label='實驗組', bins=30)
        axes[0, 1].set_title('銷量分佈對比')
        axes[0, 1].set_xlabel('銷量')
        axes[0, 1].set_ylabel('頻率')
        axes[0, 1].legend()
        
        # 3. 價格vs銷量散點圖
        treatment_data = self.data[self.data['price_treatment'] == 1]
        control_data = self.data[self.data['price_treatment'] == 0]
        
        axes[0, 2].scatter(control_data['price'], control_data['sales_volume'], 
                          alpha=0.5, label='對照組', s=10)
        axes[0, 2].scatter(treatment_data['price'], treatment_data['sales_volume'], 
                          alpha=0.5, label='實驗組', s=10)
        axes[0, 2].set_title('價格 vs 銷量')
        axes[0, 2].set_xlabel('價格')
        axes[0, 2].set_ylabel('銷量')
        axes[0, 2].legend()
        
        # 4. 傾向得分分佈
        if 'propensity_score' in self.data.columns:
            axes[1, 0].hist(self.data[self.data['price_treatment'] == 0]['propensity_score'], 
                           alpha=0.7, label='對照組', bins=30)
            axes[1, 0].hist(self.data[self.data['price_treatment'] == 1]['propensity_score'], 
                           alpha=0.7, label='實驗組', bins=30)
            axes[1, 0].set_title('傾向得分分佈')
            axes[1, 0].set_xlabel('傾向得分')
            axes[1, 0].set_ylabel('頻率')
            axes[1, 0].legend()
        
        # 5. 客戶群體銷量對比
        segment_data = self.data.groupby(['customer_segment', 'price_treatment'])['sales_volume'].mean().unstack()
        segment_data.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('不同客戶群體銷量對比')
        axes[1, 1].set_xlabel('客戶群體')
        axes[1, 1].set_ylabel('平均銷量')
        axes[1, 1].legend(['對照組', '實驗組'])
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. 處理效應總結
        effects = [
            self.results.get('experiment_analysis', {}).get('ate_sales', 0),
            self.results.get('regression_analysis', {}).get('treatment_effect', 0),
            self.results.get('propensity_score_analysis', {}).get('ps_ate', 0)
        ]
        methods = ['隨機實驗', '回歸調整', '傾向得分匹配']
        
        axes[1, 2].bar(methods, effects)
        axes[1, 2].set_title('不同方法估計的處理效應')
        axes[1, 2].set_ylabel('銷量處理效應')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """生成分析報告"""
        print("\n" + "="*60)
        print("定價策略因果推論分析報告")
        print("="*60)
        
        print(f"\n數據概況:")
        print(f"- 總樣本數: {len(self.data):,}")
        print(f"- 實驗組比例: {self.data['price_treatment'].mean():.1%}")
        print(f"- 平均價格: ${self.data['price'].mean():.2f}")
        print(f"- 平均銷量: {self.data['sales_volume'].mean():.0f}")
        
        print(f"\n主要發現:")
        if 'experiment_analysis' in self.results:
            ate = self.results['experiment_analysis']['ate_sales']
            p_val = self.results['experiment_analysis']['p_value']
            significance = "顯著" if p_val < 0.05 else "不顯著"
            print(f"- 降價策略使銷量增加 {ate:.0f} 單位 ({significance}, p={p_val:.4f})")
        
        if 'price_elasticity' in self.results:
            print(f"\n價格彈性:")
            for segment, elasticity in self.results['price_elasticity'].items():
                sensitivity = "高度敏感" if abs(elasticity) > 2 else "中度敏感" if abs(elasticity) > 1 else "低度敏感"
                print(f"- {segment}客戶對價格{sensitivity} (彈性係數: {elasticity:.3f})")
        
        print(f"\n建議:")
        print("- 基於實驗結果，適度降價策略可以有效提升銷量")
        print("- 不同客戶群體對價格敏感度不同，建議實施差異化定價")
        print("- 持續監控競爭對手價格變動，及時調整策略")
        print("- 考慮季節性因素對定價策略的影響")

def main():
    """主函數"""
    print("開始定價策略因果推論分析...")
    
    # 創建分析實例
    analyzer = PricingCausalAnalysis()
    
    # 生成數據
    print("生成模擬數據...")
    data = analyzer.generate_synthetic_data()
    print(f"數據生成完成，共 {len(data)} 條記錄")
    
    # 執行各種分析
    analyzer.naive_correlation_analysis()
    analyzer.randomized_experiment_analysis()
    analyzer.regression_adjustment_analysis()
    analyzer.propensity_score_analysis()
    analyzer.price_elasticity_analysis()
    
    # 可視化結果
    analyzer.visualize_results()
    
    # 生成報告
    analyzer.generate_report()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()