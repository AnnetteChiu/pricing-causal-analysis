"""
真實數據定價策略因果推論分析
Real Data Pricing Strategy Causal Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pricing_causal_analysis import PricingCausalAnalysis
from real_data_loader import RealDataLoader, create_sample_real_data

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_real_data(data_source, column_mapping=None):
    """
    分析真實數據的完整流程
    
    Parameters:
    -----------
    data_source : str or pd.DataFrame
        數據源
    column_mapping : dict, optional
        列名映射字典
    """
    
    print("🚀 開始真實數據定價策略因果推論分析")
    print("=" * 60)
    
    # 第一步：數據加載和預處理
    print("\n📊 第一步：數據加載和預處理")
    print("-" * 30)
    
    loader = RealDataLoader()
    
    # 加載數據
    if isinstance(data_source, str):
        if data_source.endswith('.csv'):
            raw_data = loader.load_csv_data(data_source)
        elif data_source.endswith(('.xlsx', '.xls')):
            raw_data = loader.load_excel_data(data_source)
        else:
            raise ValueError("支持的文件格式: .csv, .xlsx, .xls")
    else:
        raw_data = data_source
        loader.data = raw_data
        loader._analyze_data_structure()
    
    # 建議列名映射
    if column_mapping is None:
        print("\n💡 建議的列名映射:")
        suggestions = loader.suggest_column_mapping()
        for key, values in suggestions.items():
            if values:
                print(f"   {key}: {values}")
        
        # 自動選擇最可能的列名
        auto_mapping = {}
        for key, values in suggestions.items():
            if values:
                auto_mapping[values[0]] = key
        
        if auto_mapping:
            print(f"\n🤖 自動選擇的映射: {auto_mapping}")
            column_mapping = auto_mapping
        else:
            print("⚠️  無法自動識別列名，請手動指定column_mapping參數")
            return None
    
    # 映射列名
    if column_mapping:
        mapped_data = loader.map_columns(column_mapping)
    else:
        mapped_data = raw_data
    
    # 數據預處理
    processed_data = loader.preprocess_data(
        price_col='price',
        volume_col='sales_volume',
        date_col='date' if 'date' in mapped_data.columns else None,
        customer_col='customer_id' if 'customer_id' in mapped_data.columns else None,
        remove_outliers=True
    )
    
    # 創建處理變數
    if 'price_treatment' not in processed_data.columns:
        print("\n🎯 創建處理變數...")
        final_data = loader.create_treatment_variable(
            method='median_split'  # 基於價格中位數分組
        )
    else:
        final_data = processed_data
        print("✅ 數據中已包含處理變數")
    
    # 添加控制變數
    enhanced_data = loader.add_control_variables(
        date_col='date' if 'date' in final_data.columns else None,
        customer_col='customer_id' if 'customer_id' in final_data.columns else None,
        product_col='product_id' if 'product_id' in final_data.columns else None
    )
    
    # 第二步：因果推論分析
    print("\n🔬 第二步：因果推論分析")
    print("-" * 30)
    
    # 創建分析器並加載數據
    analyzer = PricingCausalAnalysis(data=enhanced_data)
    
    # 執行各種分析
    print("執行隨機實驗分析...")
    analyzer.randomized_experiment_analysis()
    
    print("執行回歸調整分析...")
    analyzer.regression_adjustment_analysis()
    
    print("執行傾向得分分析...")
    analyzer.propensity_score_analysis()
    
    if len(enhanced_data['customer_segment'].unique()) > 1:
        print("執行價格彈性分析...")
        analyzer.price_elasticity_analysis()
    
    # 第三步：結果可視化
    print("\n📈 第三步：結果可視化")
    print("-" * 30)
    
    create_real_data_visualization(analyzer, enhanced_data)
    
    # 第四步：生成報告
    print("\n📋 第四步：分析報告")
    print("-" * 30)
    
    generate_real_data_report(analyzer, enhanced_data, loader)
    
    return analyzer, enhanced_data

def create_real_data_visualization(analyzer, data):
    """創建真實數據的可視化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('真實數據定價策略因果推論分析結果', fontsize=16)
    
    # 1. 價格分佈對比
    treatment_data = data[data['price_treatment'] == 1]
    control_data = data[data['price_treatment'] == 0]
    
    axes[0, 0].hist(control_data['price'], alpha=0.7, label='對照組', bins=30, color='lightblue')
    axes[0, 0].hist(treatment_data['price'], alpha=0.7, label='實驗組', bins=30, color='lightcoral')
    axes[0, 0].set_title('價格分佈對比')
    axes[0, 0].set_xlabel('價格')
    axes[0, 0].set_ylabel('頻率')
    axes[0, 0].legend()
    
    # 2. 銷量分佈對比
    axes[0, 1].hist(control_data['sales_volume'], alpha=0.7, label='對照組', bins=30, color='lightblue')
    axes[0, 1].hist(treatment_data['sales_volume'], alpha=0.7, label='實驗組', bins=30, color='lightcoral')
    axes[0, 1].set_title('銷量分佈對比')
    axes[0, 1].set_xlabel('銷量')
    axes[0, 1].set_ylabel('頻率')
    axes[0, 1].legend()
    
    # 3. 價格vs銷量散點圖
    axes[0, 2].scatter(control_data['price'], control_data['sales_volume'], 
                      alpha=0.5, label='對照組', s=10, color='blue')
    axes[0, 2].scatter(treatment_data['price'], treatment_data['sales_volume'], 
                      alpha=0.5, label='實驗組', s=10, color='red')
    axes[0, 2].set_title('價格 vs 銷量')
    axes[0, 2].set_xlabel('價格')
    axes[0, 2].set_ylabel('銷量')
    axes[0, 2].legend()
    
    # 4. 處理效應比較
    methods = ['隨機實驗', '回歸調整', '傾向得分匹配']
    effects = [
        analyzer.results.get('experiment_analysis', {}).get('ate_sales', 0),
        analyzer.results.get('regression_analysis', {}).get('treatment_effect', 0),
        analyzer.results.get('propensity_score_analysis', {}).get('ps_ate', 0)
    ]
    
    bars = axes[1, 0].bar(methods, effects, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1, 0].set_title('不同方法的處理效應估計')
    axes[1, 0].set_ylabel('銷量處理效應')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 添加數值標籤
    for bar, effect in zip(bars, effects):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(effects)*0.01,
                       f'{effect:.0f}', ha='center', va='bottom')
    
    # 5. 時間趨勢（如果有日期數據）
    if 'date' in data.columns:
        daily_sales = data.groupby(['date', 'price_treatment'])['sales_volume'].mean().unstack()
        if daily_sales.shape[0] > 1:  # 確保有多個時間點
            daily_sales.plot(ax=axes[1, 1], title='銷量時間趨勢')
            axes[1, 1].set_xlabel('日期')
            axes[1, 1].set_ylabel('平均銷量')
            axes[1, 1].legend(['對照組', '實驗組'])
        else:
            axes[1, 1].text(0.5, 0.5, '數據時間跨度不足\n無法顯示趨勢', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('銷量時間趨勢')
    else:
        axes[1, 1].text(0.5, 0.5, '無日期數據', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('銷量時間趨勢')
    
    # 6. 客戶群體分析（如果有客戶分群數據）
    if 'customer_segment' in data.columns and len(data['customer_segment'].unique()) > 1:
        segment_data = data.groupby(['customer_segment', 'price_treatment'])['sales_volume'].mean().unstack()
        segment_data.plot(kind='bar', ax=axes[1, 2])
        axes[1, 2].set_title('不同客戶群體銷量對比')
        axes[1, 2].set_xlabel('客戶群體')
        axes[1, 2].set_ylabel('平均銷量')
        axes[1, 2].legend(['對照組', '實驗組'])
        axes[1, 2].tick_params(axis='x', rotation=45)
    else:
        axes[1, 2].text(0.5, 0.5, '無客戶分群數據', ha='center', va='center', 
                       transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('客戶群體分析')
    
    plt.tight_layout()
    plt.show()

def generate_real_data_report(analyzer, data, loader):
    """生成真實數據分析報告"""
    
    print("\n" + "="*60)
    print("真實數據定價策略因果推論分析報告")
    print("="*60)
    
    # 數據概況
    print(f"\n📊 數據概況:")
    print(f"   - 總樣本數: {len(data):,}")
    print(f"   - 時間跨度: {data['date'].min()} 到 {data['date'].max()}" if 'date' in data.columns else "   - 無時間信息")
    print(f"   - 實驗組比例: {data['price_treatment'].mean():.1%}")
    print(f"   - 平均價格: ${data['price'].mean():.2f}")
    print(f"   - 平均銷量: {data['sales_volume'].mean():.0f}")
    
    # 主要發現
    print(f"\n🔍 主要發現:")
    if 'experiment_analysis' in analyzer.results:
        ate = analyzer.results['experiment_analysis']['ate_sales']
        p_val = analyzer.results['experiment_analysis']['p_value']
        significance = "顯著" if p_val < 0.05 else "不顯著"
        
        price_diff = data[data['price_treatment']==1]['price'].mean() - data[data['price_treatment']==0]['price'].mean()
        if price_diff < 0:
            print(f"   - 降價策略使銷量增加 {ate:.0f} 單位 ({significance}, p={p_val:.4f})")
        else:
            print(f"   - 提價策略使銷量變化 {ate:.0f} 單位 ({significance}, p={p_val:.4f})")
    
    # 價格彈性分析
    if 'price_elasticity' in analyzer.results:
        print(f"\n💰 價格彈性分析:")
        for segment, elasticity in analyzer.results['price_elasticity'].items():
            sensitivity = "高度敏感" if abs(elasticity) > 2 else "中度敏感" if abs(elasticity) > 1 else "低度敏感"
            print(f"   - {segment}: {sensitivity} (彈性係數: {elasticity:.3f})")
    
    # 數據質量評估
    print(f"\n📋 數據質量評估:")
    missing_rate = data.isnull().sum().sum() / (len(data) * len(data.columns))
    print(f"   - 缺失值比例: {missing_rate:.2%}")
    
    if 'customer_id' in data.columns:
        repeat_customers = data['customer_id'].value_counts()
        print(f"   - 重複客戶比例: {(repeat_customers > 1).mean():.1%}")
    
    if 'date' in data.columns:
        date_range = (data['date'].max() - data['date'].min()).days
        print(f"   - 數據時間跨度: {date_range} 天")
    
    # 業務建議
    print(f"\n💡 業務建議:")
    
    # 基於處理效應的建議
    if 'experiment_analysis' in analyzer.results:
        ate = analyzer.results['experiment_analysis']['ate_sales']
        p_val = analyzer.results['experiment_analysis']['p_value']
        
        if p_val < 0.05:
            if ate > 0:
                print("   - ✅ 當前定價策略有效，建議繼續執行")
                print("   - 📈 可以考慮擴大實施範圍")
            else:
                print("   - ⚠️  當前定價策略可能不利於銷量")
                print("   - 🔄 建議重新評估定價策略")
        else:
            print("   - 📊 定價效果不明顯，需要更多數據或調整策略")
    
    # 基於價格彈性的建議
    if 'price_elasticity' in analyzer.results:
        avg_elasticity = np.mean(list(analyzer.results['price_elasticity'].values()))
        if abs(avg_elasticity) < 1:
            print("   - 💎 客戶對價格不敏感，有提價空間")
        else:
            print("   - ⚡ 客戶對價格敏感，需謹慎調價")
    
    print(f"\n📈 後續建議:")
    print("   - 持續收集更多數據以提高分析精度")
    print("   - 考慮進行更長期的跟蹤分析")
    print("   - 結合外部因素（競爭、季節性等）進行深入分析")
    print("   - 建立定期的定價策略評估機制")

def demo_with_sample_data():
    """使用示例數據進行演示"""
    
    print("🎯 創建示例數據進行演示...")
    
    # 創建示例數據
    sample_data = create_sample_real_data()
    
    # 保存示例數據
    sample_data.to_csv('sample_pricing_data.csv', index=False, encoding='utf-8-sig')
    print("✅ 示例數據已保存到 sample_pricing_data.csv")
    
    # 定義列名映射
    column_mapping = {
        '銷售價格': 'price',
        '銷售數量': 'sales_volume',
        '訂單日期': 'date',
        '客戶ID': 'customer_id',
        '產品ID': 'product_id'
    }
    
    # 執行分析
    analyzer, processed_data = analyze_real_data('sample_pricing_data.csv', column_mapping)
    
    return analyzer, processed_data

if __name__ == "__main__":
    # 演示分析流程
    try:
        analyzer, data = demo_with_sample_data()
        print("\n🎉 真實數據分析演示完成！")
        
    except Exception as e:
        print(f"❌ 分析過程中出現錯誤: {e}")
        print("請檢查數據格式和列名映射是否正確")