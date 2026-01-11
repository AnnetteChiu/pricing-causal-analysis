"""
真實數據加載器
Real Data Loader for Pricing Analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
import warnings
from pathlib import Path

class RealDataLoader:
    """真實數據加載和預處理類"""
    
    def __init__(self):
        self.data = None
        self.data_info = {}
        self.required_columns = {
            'price': '價格',
            'sales_volume': '銷量',
            'date': '日期',
            'customer_id': '客戶ID（可選）',
            'product_id': '產品ID（可選）'
        }
    
    def load_csv_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        從CSV文件加載數據
        
        Parameters:
        -----------
        file_path : str
            CSV文件路徑
        **kwargs : dict
            pandas.read_csv的其他參數
        
        Returns:
        --------
        pd.DataFrame
            加載的數據
        """
        try:
            # 嘗試不同的編碼
            encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'latin1']
            
            for encoding in encodings:
                try:
                    data = pd.read_csv(file_path, encoding=encoding, **kwargs)
                    print(f"✅ 成功使用 {encoding} 編碼加載數據")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("無法使用常見編碼讀取文件，請檢查文件格式")
            
            self.data = data
            self._analyze_data_structure()
            return data
            
        except Exception as e:
            print(f"❌ 加載數據失敗: {e}")
            raise
    
    def load_excel_data(self, file_path: str, sheet_name: Union[str, int] = 0, **kwargs) -> pd.DataFrame:
        """
        從Excel文件加載數據
        
        Parameters:
        -----------
        file_path : str
            Excel文件路徑
        sheet_name : str or int
            工作表名稱或索引
        **kwargs : dict
            pandas.read_excel的其他參數
        
        Returns:
        --------
        pd.DataFrame
            加載的數據
        """
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            self.data = data
            self._analyze_data_structure()
            print(f"✅ 成功從Excel加載數據，工作表: {sheet_name}")
            return data
            
        except Exception as e:
            print(f"❌ 加載Excel數據失敗: {e}")
            raise
    
    def _analyze_data_structure(self):
        """分析數據結構"""
        if self.data is None:
            return
        
        self.data_info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': dict(self.data.dtypes),
            'missing_values': dict(self.data.isnull().sum()),
            'numeric_columns': list(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.data.select_dtypes(include=['object']).columns),
            'date_columns': []
        }
        
        # 檢測可能的日期列
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['date', '日期', 'time', '時間']):
                self.data_info['date_columns'].append(col)
        
        print(f"📊 數據概況:")
        print(f"   - 形狀: {self.data_info['shape']}")
        print(f"   - 數值列: {len(self.data_info['numeric_columns'])} 個")
        print(f"   - 分類列: {len(self.data_info['categorical_columns'])} 個")
        print(f"   - 缺失值: {sum(self.data_info['missing_values'].values())} 個")
    
    def suggest_column_mapping(self) -> Dict[str, List[str]]:
        """建議列名映射"""
        suggestions = {}
        
        # 價格相關列
        price_keywords = ['price', '價格', 'cost', '成本', 'amount', '金額', '單價']
        suggestions['price'] = [col for col in self.data.columns 
                               if any(keyword in col.lower() for keyword in price_keywords)]
        
        # 銷量相關列
        volume_keywords = ['volume', '銷量', 'quantity', '數量', 'sales', '銷售', 'sold', '售出']
        suggestions['sales_volume'] = [col for col in self.data.columns 
                                     if any(keyword in col.lower() for keyword in volume_keywords)]
        
        # 日期相關列
        date_keywords = ['date', '日期', 'time', '時間', 'day', '天']
        suggestions['date'] = [col for col in self.data.columns 
                              if any(keyword in col.lower() for keyword in date_keywords)]
        
        # 客戶相關列
        customer_keywords = ['customer', '客戶', 'user', '用戶', 'client', '客戶端']
        suggestions['customer_id'] = [col for col in self.data.columns 
                                    if any(keyword in col.lower() for keyword in customer_keywords)]
        
        # 產品相關列
        product_keywords = ['product', '產品', 'item', '商品', 'sku', 'goods']
        suggestions['product_id'] = [col for col in self.data.columns 
                                   if any(keyword in col.lower() for keyword in product_keywords)]
        
        return suggestions
    
    def map_columns(self, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        映射列名到標準格式
        
        Parameters:
        -----------
        column_mapping : dict
            列名映射字典，例如 {'原列名': '標準列名'}
        
        Returns:
        --------
        pd.DataFrame
            映射後的數據
        """
        if self.data is None:
            raise ValueError("請先加載數據")
        
        # 創建數據副本
        mapped_data = self.data.copy()
        
        # 重命名列
        mapped_data = mapped_data.rename(columns=column_mapping)
        
        # 檢查必需的列
        missing_columns = []
        for col in ['price', 'sales_volume']:
            if col not in mapped_data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            print(f"⚠️  缺少必需的列: {missing_columns}")
            print("請確保數據包含價格和銷量信息")
        
        self.data = mapped_data
        print(f"✅ 列名映射完成")
        return mapped_data
    
    def preprocess_data(self, 
                       price_col: str = 'price',
                       volume_col: str = 'sales_volume',
                       date_col: Optional[str] = None,
                       customer_col: Optional[str] = None,
                       remove_outliers: bool = True,
                       outlier_method: str = 'iqr') -> pd.DataFrame:
        """
        數據預處理
        
        Parameters:
        -----------
        price_col : str
            價格列名
        volume_col : str
            銷量列名
        date_col : str, optional
            日期列名
        customer_col : str, optional
            客戶列名
        remove_outliers : bool
            是否移除異常值
        outlier_method : str
            異常值檢測方法 ('iqr' 或 'zscore')
        
        Returns:
        --------
        pd.DataFrame
            預處理後的數據
        """
        if self.data is None:
            raise ValueError("請先加載數據")
        
        processed_data = self.data.copy()
        
        # 1. 處理缺失值
        print("🔧 處理缺失值...")
        initial_rows = len(processed_data)
        processed_data = processed_data.dropna(subset=[price_col, volume_col])
        removed_rows = initial_rows - len(processed_data)
        if removed_rows > 0:
            print(f"   移除了 {removed_rows} 行包含缺失值的數據")
        
        # 2. 數據類型轉換
        print("🔧 轉換數據類型...")
        processed_data[price_col] = pd.to_numeric(processed_data[price_col], errors='coerce')
        processed_data[volume_col] = pd.to_numeric(processed_data[volume_col], errors='coerce')
        
        # 3. 處理日期列
        if date_col and date_col in processed_data.columns:
            print("🔧 處理日期數據...")
            processed_data[date_col] = pd.to_datetime(processed_data[date_col], errors='coerce')
            processed_data = processed_data.dropna(subset=[date_col])
        
        # 4. 移除異常值
        if remove_outliers:
            print(f"🔧 使用 {outlier_method} 方法移除異常值...")
            processed_data = self._remove_outliers(processed_data, [price_col, volume_col], method=outlier_method)
        
        # 5. 基本數據驗證
        print("🔧 數據驗證...")
        # 移除負價格和負銷量
        initial_rows = len(processed_data)
        processed_data = processed_data[
            (processed_data[price_col] > 0) & 
            (processed_data[volume_col] >= 0)
        ]
        removed_rows = initial_rows - len(processed_data)
        if removed_rows > 0:
            print(f"   移除了 {removed_rows} 行無效數據（負價格或負銷量）")
        
        self.data = processed_data
        print(f"✅ 數據預處理完成，最終數據形狀: {processed_data.shape}")
        
        return processed_data
    
    def _remove_outliers(self, data: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """移除異常值"""
        cleaned_data = data.copy()
        
        for col in columns:
            if method == 'iqr':
                Q1 = cleaned_data[col].quantile(0.25)
                Q3 = cleaned_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((cleaned_data[col] - cleaned_data[col].mean()) / cleaned_data[col].std())
                outliers = z_scores > 3
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"   {col}: 移除了 {outlier_count} 個異常值")
                cleaned_data = cleaned_data[~outliers]
        
        return cleaned_data
    
    def create_treatment_variable(self, 
                                method: str = 'price_change',
                                threshold: Optional[float] = None,
                                date_col: Optional[str] = None,
                                treatment_date: Optional[str] = None) -> pd.DataFrame:
        """
        創建處理變數（實驗組/對照組標識）
        
        Parameters:
        -----------
        method : str
            創建方法 ('price_change', 'median_split', 'date_based', 'random')
        threshold : float, optional
            閾值（用於price_change方法）
        date_col : str, optional
            日期列名（用於date_based方法）
        treatment_date : str, optional
            處理開始日期（用於date_based方法）
        
        Returns:
        --------
        pd.DataFrame
            包含處理變數的數據
        """
        if self.data is None:
            raise ValueError("請先加載和預處理數據")
        
        data_with_treatment = self.data.copy()
        
        if method == 'price_change':
            # 基於價格變化創建處理變數
            if 'price' not in data_with_treatment.columns:
                raise ValueError("數據中沒有找到price列")
            
            if threshold is None:
                threshold = data_with_treatment['price'].median()
            
            data_with_treatment['price_treatment'] = (data_with_treatment['price'] < threshold).astype(int)
            print(f"✅ 基於價格閾值 {threshold:.2f} 創建處理變數")
            
        elif method == 'median_split':
            # 基於價格中位數分組
            median_price = data_with_treatment['price'].median()
            data_with_treatment['price_treatment'] = (data_with_treatment['price'] < median_price).astype(int)
            print(f"✅ 基於價格中位數 {median_price:.2f} 創建處理變數")
            
        elif method == 'date_based':
            # 基於日期創建處理變數
            if date_col is None or date_col not in data_with_treatment.columns:
                raise ValueError("date_based方法需要指定有效的日期列")
            
            if treatment_date is None:
                raise ValueError("date_based方法需要指定treatment_date")
            
            treatment_date = pd.to_datetime(treatment_date)
            data_with_treatment['price_treatment'] = (data_with_treatment[date_col] >= treatment_date).astype(int)
            print(f"✅ 基於日期 {treatment_date} 創建處理變數")
            
        elif method == 'random':
            # 隨機分組
            np.random.seed(42)
            data_with_treatment['price_treatment'] = np.random.binomial(1, 0.5, len(data_with_treatment))
            print("✅ 隨機創建處理變數")
            
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 顯示分組統計
        treatment_counts = data_with_treatment['price_treatment'].value_counts()
        print(f"   處理組分佈: 對照組 {treatment_counts[0]} 個, 實驗組 {treatment_counts[1]} 個")
        
        self.data = data_with_treatment
        return data_with_treatment
    
    def add_control_variables(self, 
                            date_col: Optional[str] = None,
                            customer_col: Optional[str] = None,
                            product_col: Optional[str] = None) -> pd.DataFrame:
        """添加控制變數"""
        if self.data is None:
            raise ValueError("請先加載數據")
        
        enhanced_data = self.data.copy()
        
        # 添加時間相關變數
        if date_col and date_col in enhanced_data.columns:
            enhanced_data['year'] = enhanced_data[date_col].dt.year
            enhanced_data['month'] = enhanced_data[date_col].dt.month
            enhanced_data['quarter'] = enhanced_data[date_col].dt.quarter
            enhanced_data['weekday'] = enhanced_data[date_col].dt.weekday
            
            # 季節變數
            season_map = {12: '冬', 1: '冬', 2: '冬',
                         3: '春', 4: '春', 5: '春',
                         6: '夏', 7: '夏', 8: '夏',
                         9: '秋', 10: '秋', 11: '秋'}
            enhanced_data['season'] = enhanced_data['month'].map(season_map)
            print("✅ 添加了時間相關控制變數")
        
        # 添加客戶相關變數
        if customer_col and customer_col in enhanced_data.columns:
            # 客戶購買頻次
            customer_freq = enhanced_data[customer_col].value_counts()
            enhanced_data['customer_frequency'] = enhanced_data[customer_col].map(customer_freq)
            
            # 客戶分組（基於購買頻次）
            freq_quantiles = enhanced_data['customer_frequency'].quantile([0.33, 0.67])
            enhanced_data['customer_segment'] = pd.cut(
                enhanced_data['customer_frequency'],
                bins=[0, freq_quantiles[0.33], freq_quantiles[0.67], float('inf')],
                labels=['低端', '中端', '高端']
            )
            print("✅ 添加了客戶相關控制變數")
        
        # 添加產品相關變數
        if product_col and product_col in enhanced_data.columns:
            # 產品平均價格
            product_avg_price = enhanced_data.groupby(product_col)['price'].mean()
            enhanced_data['product_avg_price'] = enhanced_data[product_col].map(product_avg_price)
            
            # 產品價格相對位置
            enhanced_data['price_relative'] = enhanced_data['price'] / enhanced_data['product_avg_price']
            print("✅ 添加了產品相關控制變數")
        
        self.data = enhanced_data
        return enhanced_data
    
    def export_processed_data(self, file_path: str, format: str = 'csv'):
        """導出處理後的數據"""
        if self.data is None:
            raise ValueError("沒有數據可以導出")
        
        if format.lower() == 'csv':
            self.data.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif format.lower() == 'excel':
            self.data.to_excel(file_path, index=False)
        else:
            raise ValueError("支持的格式: 'csv' 或 'excel'")
        
        print(f"✅ 數據已導出到: {file_path}")
    
    def get_data_summary(self) -> Dict:
        """獲取數據摘要"""
        if self.data is None:
            return {}
        
        summary = {
            'basic_info': {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # 數值變數摘要
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': self.data[col].mean(),
                'median': self.data[col].median(),
                'std': self.data[col].std(),
                'min': self.data[col].min(),
                'max': self.data[col].max(),
                'missing': self.data[col].isnull().sum()
            }
        
        # 分類變數摘要
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': self.data[col].nunique(),
                'top_values': self.data[col].value_counts().head().to_dict(),
                'missing': self.data[col].isnull().sum()
            }
        
        return summary

def create_sample_real_data():
    """創建示例真實數據格式"""
    np.random.seed(42)
    
    # 模擬電商數據
    n_records = 5000
    start_date = pd.to_datetime('2023-01-01')
    end_date = pd.to_datetime('2023-12-31')
    
    data = {
        '訂單日期': pd.date_range(start_date, end_date, periods=n_records),
        '產品ID': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_records),
        '客戶ID': np.random.choice(range(1, 1001), n_records),
        '銷售價格': np.random.normal(100, 25, n_records),
        '銷售數量': np.random.poisson(5, n_records),
        '促銷活動': np.random.choice(['無', '滿減', '折扣', 'VIP'], n_records),
        '銷售渠道': np.random.choice(['線上', '線下', '移動端'], n_records),
        '地區': np.random.choice(['北京', '上海', '廣州', '深圳', '杭州'], n_records)
    }
    
    df = pd.DataFrame(data)
    
    # 確保價格和數量為正數
    df['銷售價格'] = np.abs(df['銷售價格'])
    df['銷售數量'] = np.abs(df['銷售數量'])
    
    # 添加一些業務邏輯
    df.loc[df['促銷活動'] == '折扣', '銷售價格'] *= 0.8
    df.loc[df['促銷活動'] == 'VIP', '銷售價格'] *= 0.9
    
    return df

if __name__ == "__main__":
    # 創建示例數據
    sample_data = create_sample_real_data()
    sample_data.to_csv('sample_pricing_data.csv', index=False, encoding='utf-8-sig')
    print("✅ 創建了示例數據文件: sample_pricing_data.csv")
    
    # 演示數據加載流程
    loader = RealDataLoader()
    
    # 加載數據
    data = loader.load_csv_data('sample_pricing_data.csv')
    
    # 建議列名映射
    suggestions = loader.suggest_column_mapping()
    print("\n📋 建議的列名映射:")
    for key, values in suggestions.items():
        if values:
            print(f"   {key}: {values}")
    
    # 映射列名
    column_mapping = {
        '銷售價格': 'price',
        '銷售數量': 'sales_volume',
        '訂單日期': 'date',
        '客戶ID': 'customer_id',
        '產品ID': 'product_id'
    }
    
    mapped_data = loader.map_columns(column_mapping)
    
    # 預處理數據
    processed_data = loader.preprocess_data(
        price_col='price',
        volume_col='sales_volume',
        date_col='date',
        customer_col='customer_id'
    )
    
    # 創建處理變數
    final_data = loader.create_treatment_variable(method='median_split')
    
    # 添加控制變數
    enhanced_data = loader.add_control_variables(
        date_col='date',
        customer_col='customer_id',
        product_col='product_id'
    )
    
    print(f"\n✅ 最終數據形狀: {enhanced_data.shape}")
    print(f"✅ 列名: {list(enhanced_data.columns)}")