"""
基礎分析模塊測試
Tests for basic causal analysis module
"""

import unittest
import numpy as np
import pandas as pd
from pricing_causal_analysis import PricingCausalAnalysis


class TestPricingCausalAnalysis(unittest.TestCase):
    """測試定價因果分析類"""
    
    def setUp(self):
        """設置測試環境"""
        self.analyzer = PricingCausalAnalysis()
        self.analyzer.generate_synthetic_data(n_samples=1000, seed=42)
    
    def test_data_generation(self):
        """測試數據生成"""
        self.assertIsNotNone(self.analyzer.data)
        self.assertEqual(len(self.analyzer.data), 1000)
        
        # 檢查必要的列是否存在
        required_columns = [
            'price', 'sales_volume', 'price_treatment', 
            'customer_segment', 'season'
        ]
        for col in required_columns:
            self.assertIn(col, self.analyzer.data.columns)
    
    def test_randomized_experiment_analysis(self):
        """測試隨機實驗分析"""
        self.analyzer.randomized_experiment_analysis()
        
        # 檢查結果是否存在
        self.assertIn('experiment_analysis', self.analyzer.results)
        results = self.analyzer.results['experiment_analysis']
        
        # 檢查關鍵指標
        self.assertIn('ate_sales', results)
        self.assertIn('p_value', results)
        self.assertIsInstance(results['ate_sales'], (int, float))
        self.assertIsInstance(results['p_value'], (int, float))
    
    def test_regression_adjustment_analysis(self):
        """測試回歸調整分析"""
        self.analyzer.regression_adjustment_analysis()
        
        # 檢查結果是否存在
        self.assertIn('regression_analysis', self.analyzer.results)
        results = self.analyzer.results['regression_analysis']
        
        # 檢查關鍵指標
        self.assertIn('treatment_effect', results)
        self.assertIn('r_squared', results)
        self.assertIsInstance(results['treatment_effect'], (int, float))
        self.assertTrue(0 <= results['r_squared'] <= 1)
    
    def test_propensity_score_analysis(self):
        """測試傾向得分分析"""
        self.analyzer.propensity_score_analysis()
        
        # 檢查結果是否存在
        self.assertIn('propensity_score_analysis', self.analyzer.results)
        results = self.analyzer.results['propensity_score_analysis']
        
        # 檢查關鍵指標
        self.assertIn('ps_ate', results)
        self.assertIn('matched_pairs', results)
        self.assertIsInstance(results['ps_ate'], (int, float))
        self.assertIsInstance(results['matched_pairs'], int)
    
    def test_price_elasticity_analysis(self):
        """測試價格彈性分析"""
        self.analyzer.price_elasticity_analysis()
        
        # 檢查結果是否存在
        self.assertIn('price_elasticity', self.analyzer.results)
        elasticities = self.analyzer.results['price_elasticity']
        
        # 檢查每個客戶群體都有彈性估計
        segments = self.analyzer.data['customer_segment'].unique()
        for segment in segments:
            self.assertIn(segment, elasticities)
            self.assertIsInstance(elasticities[segment], (int, float))
    
    def test_data_quality(self):
        """測試數據質量"""
        data = self.analyzer.data
        
        # 檢查沒有缺失值
        self.assertFalse(data.isnull().any().any())
        
        # 檢查銷量為正數
        self.assertTrue((data['sales_volume'] >= 0).all())
        
        # 檢查價格為正數
        self.assertTrue((data['price'] > 0).all())
        
        # 檢查處理變數為0或1
        self.assertTrue(data['price_treatment'].isin([0, 1]).all())
    
    def test_treatment_effect_direction(self):
        """測試處理效應方向（降價應該增加銷量）"""
        self.analyzer.randomized_experiment_analysis()
        
        ate = self.analyzer.results['experiment_analysis']['ate_sales']
        
        # 由於我們設定降價，處理效應應該為正（增加銷量）
        self.assertGreater(ate, 0, "降價策略應該增加銷量")


class TestDataValidation(unittest.TestCase):
    """測試數據驗證功能"""
    
    def test_empty_data_handling(self):
        """測試空數據處理"""
        analyzer = PricingCausalAnalysis()
        
        # 測試在沒有數據時調用分析方法
        with self.assertRaises((AttributeError, ValueError)):
            analyzer.randomized_experiment_analysis()
    
    def test_small_sample_handling(self):
        """測試小樣本處理"""
        analyzer = PricingCausalAnalysis()
        analyzer.generate_synthetic_data(n_samples=10)
        
        # 小樣本應該仍能運行，但可能結果不穩定
        try:
            analyzer.randomized_experiment_analysis()
            analyzer.regression_adjustment_analysis()
        except Exception as e:
            self.fail(f"小樣本分析失敗: {e}")


if __name__ == '__main__':
    unittest.main()