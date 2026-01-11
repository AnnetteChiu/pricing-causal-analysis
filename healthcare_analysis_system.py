"""
健保費用數據爬蟲與因果推論分析整合系統
Integrated Healthcare Cost Scraping and Causal Analysis System
"""

from healthcare_cost_scraper import HealthcareCostScraper, HealthcareCostAnalyzer
from healthcare_causal_analysis import HealthcareCausalAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def run_complete_healthcare_analysis():
    """運行完整的健保分析流程"""
    
    print("🏥 健保費用數據爬蟲與因果推論分析系統")
    print("=" * 70)
    
    # 第一階段：數據爬蟲
    print("\n📡 第一階段：健保數據爬蟲")
    print("-" * 40)
    
    scraper = HealthcareCostScraper()
    healthcare_data = scraper.run_scraping()
    
    # 保存原始數據
    df = scraper.save_data('healthcare_cost_data.csv')
    
    # 第二階段：描述性統計分析
    print("\n📊 第二階段：描述性統計分析")
    print("-" * 40)
    
    analyzer = HealthcareCostAnalyzer(healthcare_data)
    analyzer.generate_report()
    
    # 第三階段：因果推論分析
    print("\n🔬 第三階段：因果推論分析")
    print("-" * 40)
    
    causal_analyzer = HealthcareCausalAnalysis(healthcare_data)
    causal_analyzer.generate_healthcare_report()
    
    # 第四階段：綜合可視化
    print("\n📈 第四階段：綜合可視化")
    print("-" * 40)
    
    create_comprehensive_visualization(analyzer, causal_analyzer)
    
    print("\n🎉 完整分析流程結束！")
    
    return scraper, analyzer, causal_analyzer

def create_comprehensive_visualization(desc_analyzer, causal_analyzer):
    """創建綜合可視化報告"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('健保費用綜合分析報告', fontsize=16)
    
    # 1. 健保費用趨勢
    nhi_data = desc_analyzer.df[
        (desc_analyzer.df['category'] == '健保總費用') & 
        (desc_analyzer.df['data_type'] == 'nhi_statistics')
    ].sort_values('date')
    
    if not nhi_data.empty:
        axes[0, 0].plot(nhi_data['date'], nhi_data['amount']/1e9, 
                       marker='o', linewidth=2, color='blue')
        axes[0, 0].set_title('健保總費用趨勢')
        axes[0, 0].set_xlabel('時間')
        axes[0, 0].set_ylabel('費用 (億元)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 地區醫療費用比較
    price_data = desc_analyzer.df[desc_analyzer.df['data_type'] == 'medical_price']
    if not price_data.empty:
        regional_avg = price_data.groupby('region')['amount'].mean().sort_values(ascending=True)
        bars = axes[0, 1].barh(regional_avg.index, regional_avg.values, color='orange')
        axes[0, 1].set_title('各地區平均醫療費用')
        axes[0, 1].set_xlabel('費用 (元)')
        
        # 添加數值標籤
        for bar in bars:
            width = bar.get_width()
            axes[0, 1].text(width, bar.get_y() + bar.get_height()/2, 
                           f'{width:.0f}', ha='left', va='center')
    
    # 3. 醫院類型分析
    hospital_data = desc_analyzer.df[desc_analyzer.df['data_type'] == 'hospital_data']
    if not hospital_data.empty:
        bed_data = hospital_data[hospital_data['subcategory'] == '總病床數']
        if not bed_data.empty:
            type_beds = bed_data.groupby('hospital_type')['amount'].sum()
            wedges, texts, autotexts = axes[1, 0].pie(type_beds.values, 
                                                     labels=type_beds.index, 
                                                     autopct='%1.1f%%',
                                                     colors=['lightblue', 'lightcoral', 'lightgreen'])
            axes[1, 0].set_title('各類醫院病床數分布')
    
    # 4. 因果推論結果
    if causal_analyzer.causal_analyzer and causal_analyzer.causal_analyzer.results:
        results = causal_analyzer.causal_analyzer.results
        
        methods = []
        effects = []
        
        if 'experiment_analysis' in results:
            methods.append('隨機實驗')
            effects.append(results['experiment_analysis']['ate_sales'])
        
        if 'regression_analysis' in results:
            methods.append('回歸調整')
            effects.append(results['regression_analysis']['treatment_effect'])
        
        if methods and effects:
            bars = axes[1, 1].bar(methods, effects, color=['skyblue', 'lightcoral'])
            axes[1, 1].set_title('政策影響效果 (因果推論)')
            axes[1, 1].set_ylabel('就診次數變化')
            
            # 添加數值標籤
            for bar, effect in zip(bars, effects):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, height,
                               f'{effect:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def create_policy_impact_simulation():
    """創建政策影響模擬"""
    
    print("\n🎯 健保政策影響模擬分析")
    print("-" * 40)
    
    # 模擬不同政策情境
    scenarios = {
        '現狀維持': {'cost_change': 0, 'access_change': 0},
        '提高給付': {'cost_change': 0.1, 'access_change': 0.15},
        '降低給付': {'cost_change': -0.05, 'access_change': -0.1},
        '分級醫療': {'cost_change': -0.08, 'access_change': 0.05},
        '預防保健': {'cost_change': -0.12, 'access_change': 0.2}
    }
    
    # 基準值
    base_cost = 500  # 億元
    base_access = 1000  # 萬人次
    
    results = {}
    for scenario, changes in scenarios.items():
        new_cost = base_cost * (1 + changes['cost_change'])
        new_access = base_access * (1 + changes['access_change'])
        
        results[scenario] = {
            'cost': new_cost,
            'access': new_access,
            'efficiency': new_access / new_cost
        }
        
        print(f"{scenario}:")
        print(f"  預估費用: {new_cost:.1f} 億元 ({changes['cost_change']:+.1%})")
        print(f"  就醫人次: {new_access:.1f} 萬人次 ({changes['access_change']:+.1%})")
        print(f"  效率指標: {new_access/new_cost:.2f} 萬人次/億元")
        print()
    
    # 可視化政策比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios_list = list(results.keys())
    costs = [results[s]['cost'] for s in scenarios_list]
    access = [results[s]['access'] for s in scenarios_list]
    
    # 費用比較
    bars1 = ax1.bar(scenarios_list, costs, color='lightcoral')
    ax1.set_title('各政策情境預估費用')
    ax1.set_ylabel('費用 (億元)')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, cost in zip(bars1, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{cost:.1f}', ha='center', va='bottom')
    
    # 就醫人次比較
    bars2 = ax2.bar(scenarios_list, access, color='lightblue')
    ax2.set_title('各政策情境預估就醫人次')
    ax2.set_ylabel('就醫人次 (萬人次)')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, acc in zip(bars2, access):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'{acc:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # 運行完整分析
    scraper, desc_analyzer, causal_analyzer = run_complete_healthcare_analysis()
    
    # 政策影響模擬
    policy_results = create_policy_impact_simulation()
    
    print("\n📋 分析總結:")
    print("✅ 完成健保數據爬蟲")
    print("✅ 完成描述性統計分析") 
    print("✅ 完成因果推論分析")
    print("✅ 完成政策影響模擬")
    print("✅ 生成綜合可視化報告")