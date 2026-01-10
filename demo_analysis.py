"""
定價策略因果推論分析演示
Pricing Strategy Causal Analysis Demo
"""

from pricing_causal_analysis import PricingCausalAnalysis
import matplotlib.pyplot as plt
import numpy as np

def run_demo():
    """運行演示分析"""
    print("=" * 60)
    print("定價策略因果推論分析演示")
    print("=" * 60)
    
    # 創建分析實例
    analyzer = PricingCausalAnalysis()
    
    # 生成數據
    print("\n1. 生成模擬數據...")
    data = analyzer.generate_synthetic_data(n_samples=5000)
    print(f"   數據生成完成，共 {len(data):,} 條記錄")
    
    # 基礎分析
    print("\n2. 執行因果推論分析...")
    analyzer.naive_correlation_analysis()
    analyzer.randomized_experiment_analysis()
    analyzer.regression_adjustment_analysis()
    analyzer.propensity_score_analysis()
    analyzer.price_elasticity_analysis()
    
    # 生成報告
    print("\n3. 生成分析報告...")
    analyzer.generate_report()
    
    # 創建簡化的可視化
    create_summary_visualization(analyzer)
    
    return analyzer

def create_summary_visualization(analyzer):
    """創建總結可視化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('定價策略因果推論分析結果總結', fontsize=16)
    
    # 1. 處理效應比較
    methods = ['隨機實驗', '回歸調整', '傾向得分匹配']
    effects = [
        analyzer.results.get('experiment_analysis', {}).get('ate_sales', 0),
        analyzer.results.get('regression_analysis', {}).get('treatment_effect', 0),
        analyzer.results.get('propensity_score_analysis', {}).get('ps_ate', 0)
    ]
    
    bars = axes[0, 0].bar(methods, effects, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('不同方法的處理效應估計')
    axes[0, 0].set_ylabel('銷量處理效應')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 添加真實值線
    axes[0, 0].axhline(y=150, color='red', linestyle='--', label='真實效應 (150)')
    axes[0, 0].legend()
    
    # 添加數值標籤
    for bar, effect in zip(bars, effects):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{effect:.0f}', ha='center', va='bottom')
    
    # 2. 價格彈性比較
    if 'price_elasticity' in analyzer.results:
        segments = list(analyzer.results['price_elasticity'].keys())
        elasticities = list(analyzer.results['price_elasticity'].values())
        
        bars = axes[0, 1].bar(segments, elasticities, color=['gold', 'orange', 'tomato'])
        axes[0, 1].set_title('不同客戶群體的價格彈性')
        axes[0, 1].set_ylabel('價格彈性係數')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, elasticity in zip(bars, elasticities):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
                           f'{elasticity:.3f}', ha='center', va='top')
    
    # 3. 價格分佈對比
    treatment_data = analyzer.data[analyzer.data['price_treatment'] == 1]
    control_data = analyzer.data[analyzer.data['price_treatment'] == 0]
    
    axes[1, 0].hist(control_data['price'], alpha=0.7, label='對照組', bins=30, color='lightblue')
    axes[1, 0].hist(treatment_data['price'], alpha=0.7, label='實驗組', bins=30, color='lightcoral')
    axes[1, 0].set_title('價格分佈對比')
    axes[1, 0].set_xlabel('價格')
    axes[1, 0].set_ylabel('頻率')
    axes[1, 0].legend()
    
    # 4. 銷量分佈對比
    axes[1, 1].hist(control_data['sales_volume'], alpha=0.7, label='對照組', bins=30, color='lightblue')
    axes[1, 1].hist(treatment_data['sales_volume'], alpha=0.7, label='實驗組', bins=30, color='lightcoral')
    axes[1, 1].set_title('銷量分佈對比')
    axes[1, 1].set_xlabel('銷量')
    axes[1, 1].set_ylabel('頻率')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印關鍵洞察
    print("\n" + "="*60)
    print("關鍵洞察與建議")
    print("="*60)
    
    ate = analyzer.results.get('experiment_analysis', {}).get('ate_sales', 0)
    p_val = analyzer.results.get('experiment_analysis', {}).get('p_value', 1)
    
    print(f"\n📊 實驗結果:")
    print(f"   • 降價策略使銷量增加 {ate:.0f} 單位")
    print(f"   • 統計顯著性: {'顯著' if p_val < 0.05 else '不顯著'} (p={p_val:.4f})")
    
    if 'price_elasticity' in analyzer.results:
        print(f"\n💰 價格敏感度:")
        for segment, elasticity in analyzer.results['price_elasticity'].items():
            sensitivity = "高度敏感" if abs(elasticity) > 2 else "中度敏感" if abs(elasticity) > 1 else "低度敏感"
            print(f"   • {segment}客戶: {sensitivity} (彈性={elasticity:.3f})")
    
    print(f"\n🎯 策略建議:")
    print("   • 適度降價可以有效提升銷量和市場份額")
    print("   • 不同客戶群體實施差異化定價策略")
    print("   • 持續監控競爭對手價格變動")
    print("   • 考慮季節性和庫存因素的影響")
    
    print(f"\n⚠️  注意事項:")
    print("   • 短期銷量增加可能影響長期利潤")
    print("   • 需要評估價格戰的風險")
    print("   • 建議進行更長期的跟蹤分析")

if __name__ == "__main__":
    analyzer = run_demo()