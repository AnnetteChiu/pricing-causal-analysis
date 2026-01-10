"""
運行完整的定價策略因果推論分析
Complete Pricing Strategy Causal Analysis Runner
"""

from pricing_causal_analysis import PricingCausalAnalysis, main as run_basic_analysis
from advanced_pricing_methods import run_advanced_analysis
import matplotlib.pyplot as plt

def main():
    """運行完整分析流程"""
    print("=" * 80)
    print("定價策略因果推論分析系統")
    print("Pricing Strategy Causal Inference Analysis System")
    print("=" * 80)
    
    # 第一步：運行基礎分析
    print("\n第一階段：基礎因果推論分析")
    print("-" * 40)
    basic_analyzer = run_basic_analysis()
    
    # 第二步：運行進階分析
    print("\n第二階段：進階因果推論分析")
    print("-" * 40)
    advanced_analyzer = run_advanced_analysis(basic_analyzer)
    
    # 第三步：綜合比較
    print("\n第三階段：方法綜合比較")
    print("-" * 40)
    compare_methods(basic_analyzer, advanced_analyzer)
    
    print("\n分析完成！")
    return basic_analyzer, advanced_analyzer

def compare_methods(basic_analyzer, advanced_analyzer):
    """比較不同方法的結果"""
    
    # 收集所有方法的處理效應估計
    effects = {}
    
    # 基礎方法
    if 'experiment_analysis' in basic_analyzer.results:
        effects['隨機實驗 (A/B測試)'] = basic_analyzer.results['experiment_analysis']['ate_sales']
    
    if 'regression_analysis' in basic_analyzer.results:
        effects['回歸調整'] = basic_analyzer.results['regression_analysis']['treatment_effect']
    
    if 'propensity_score_analysis' in basic_analyzer.results:
        effects['傾向得分匹配'] = basic_analyzer.results['propensity_score_analysis']['ps_ate']
    
    # 進階方法
    if 'did_analysis' in advanced_analyzer.results:
        effects['差分差分法'] = advanced_analyzer.results['did_analysis']['did_effect']
    
    if 'synthetic_control' in advanced_analyzer.results:
        effects['合成控制法'] = advanced_analyzer.results['synthetic_control']['treatment_effect']
    
    if 'ml_causal' in advanced_analyzer.results:
        effects['Double ML'] = advanced_analyzer.results['ml_causal']['dml_effect']
    
    # 可視化比較
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 方法比較圖
    methods = list(effects.keys())
    values = list(effects.values())
    colors = plt.cm.Set3(range(len(methods)))
    
    bars = ax1.bar(methods, values, color=colors)
    ax1.set_title('不同因果推論方法的處理效應估計', fontsize=14)
    ax1.set_ylabel('銷量處理效應')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加數值標籤
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{value:.0f}', ha='center', va='bottom')
    
    # 方法特點比較
    method_characteristics = {
        '隨機實驗': {'內部效度': 5, '外部效度': 3, '實施難度': 4, '成本': 4},
        '回歸調整': {'內部效度': 3, '外部效度': 4, '實施難度': 2, '成本': 1},
        '傾向得分': {'內部效度': 3, '外部效度': 3, '實施難度': 3, '成本': 2},
        '差分差分': {'內部效度': 4, '外部效度': 4, '實施難度': 3, '成本': 2},
        '工具變數': {'內部效度': 4, '外部效度': 3, '實施難度': 4, '成本': 2},
        '合成控制': {'內部效度': 4, '外部效度': 3, '實施難度': 4, '成本': 3},
        'Double ML': {'內部效度': 4, '外部效度': 4, '實施難度': 5, '成本': 3}
    }
    
    # 雷達圖
    categories = list(next(iter(method_characteristics.values())).keys())
    angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
    angles += angles[:1]
    
    ax2 = plt.subplot(122, projection='polar')
    
    colors_radar = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (method, scores) in enumerate(method_characteristics.items()):
        values_radar = list(scores.values())
        values_radar += values_radar[:1]
        
        ax2.plot(angles, values_radar, 'o-', linewidth=2, 
                label=method, color=colors_radar[i % len(colors_radar)])
        ax2.fill(angles, values_radar, alpha=0.1, color=colors_radar[i % len(colors_radar)])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_ylim(0, 5)
    ax2.set_title('方法特點比較', y=1.08)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.show()
    
    # 打印總結
    print("\n方法選擇建議:")
    print("=" * 50)
    print("1. 隨機實驗 (A/B測試):")
    print("   - 優點: 因果識別最可靠，內部效度高")
    print("   - 缺點: 實施成本高，可能影響用戶體驗")
    print("   - 適用: 有實驗條件的線上平台")
    
    print("\n2. 回歸調整:")
    print("   - 優點: 實施簡單，成本低")
    print("   - 缺點: 依賴無遺漏變數假設")
    print("   - 適用: 觀察性數據分析的起點")
    
    print("\n3. 差分差分法:")
    print("   - 優點: 控制時間不變的混淆因子")
    print("   - 缺點: 需要平行趨勢假設")
    print("   - 適用: 有政策變化或自然實驗的情況")
    
    print("\n4. 機器學習方法:")
    print("   - 優點: 處理高維數據和非線性關係")
    print("   - 缺點: 解釋性較差，需要大量數據")
    print("   - 適用: 大數據環境下的複雜分析")
    
    print(f"\n真實處理效應 (模擬設定): 150")
    print("最接近真實值的方法:")
    
    true_effect = 150
    best_method = min(effects.items(), key=lambda x: abs(x[1] - true_effect))
    print(f"- {best_method[0]}: {best_method[1]:.0f} (誤差: {abs(best_method[1] - true_effect):.0f})")

if __name__ == "__main__":
    basic_analyzer, advanced_analyzer = main()