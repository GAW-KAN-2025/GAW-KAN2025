import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re

# 定义不同的seq和pre长度组合
seq_lengths = [4, 8, 12]
pre_lengths = [4, 8, 12]

# 获取所有相关csv文件
result_dir = 'results'

# 处理每个seq和pre组合
for seq_len in seq_lengths:
    for pre_len in pre_lengths:
        print(f"\n=== Processing seq{seq_len}_pre{pre_len} ===")
        
        # 构建文件匹配模式
        pattern = os.path.join(result_dir, f'*_PEMS-BAY_seq{seq_len}_pre{pre_len}bs512_seed*.csv')
        files = [f for f in glob.glob(pattern) if any(f'seed{year}' in f for year in range(2021, 2031))]
        
        if not files:
            print(f"No files found for seq{seq_len}_pre{pre_len}")
            continue
            
        # 从文件名中提取参数信息
        sample_filename = os.path.basename(files[0])
        dataset_match = re.search(r'([A-Z-]+)_seq', sample_filename)
        dataset_name = dataset_match.group(1) if dataset_match else 'unknown'
        
        # 按模型分组
        model_groups = {}
        for f in files:
            basename = os.path.basename(f)
            model = basename.split('_')[0]
            model_groups.setdefault(model, []).append(f)
        
        required_seeds = {str(year) for year in range(2021, 2031)}
        
        plt.figure(figsize=(10, 5))
        
        # 用于存储所有模型的mean_mse数据
        all_models_data = {}
        
        for model, model_files in model_groups.items():
            # 排除GAF和VAR模型
            if model in ['GAF', 'VAR']:
                continue
            # 检查该模型的文件是否包含所有所需seed
            seeds_in_files = {re.search(r'seed(\d+)', os.path.basename(f)).group(1) for f in model_files if re.search(r'seed(\d+)', os.path.basename(f))}
            if not required_seeds.issubset(seeds_in_files):
                continue  # 跳过seed不全的模型
            all_mse = []
            for f in model_files:
                df = pd.read_csv(f)
                all_mse.append(df['MSE'].values)
            all_mse = np.array(all_mse)
            mean_mse = np.mean(all_mse, axis=0)
            std_mse = np.std(all_mse, axis=0, ddof=1)
            stderr_mse = std_mse / np.sqrt(all_mse.shape[0])
            horizons = df['horizon'].values
            lower = mean_mse - 1.96 * stderr_mse
            upper = mean_mse + 1.96 * stderr_mse
            # 新增打印统计信息
            ci_90 = 1.645 * stderr_mse  # 90% 置信区间
            ci_99 = 2.576 * stderr_mse  # 99% 置信区间
            # 只输出整体均值
            print(f"Model: {model}")
            print(f"Mean MSE: {np.mean(mean_mse):.6f}")
            print(f"Standard Deviation: {np.mean(std_mse):.6f}")
            print(f"90% Confidence Interval: [{np.mean(mean_mse - ci_90):.6f}, {np.mean(mean_mse + ci_90):.6f}]")
            print(f"99% Confidence Interval: [{np.mean(mean_mse - ci_99):.6f}, {np.mean(mean_mse + ci_99):.6f}]")
            
            # 存储该模型的mean_mse数据
            all_models_data[model] = mean_mse
            
            plt.plot(horizons, mean_mse, label=model)
            plt.fill_between(horizons, lower, upper, alpha=0.15)
        
        # 保存所有模型的mean_mse为CSV文件
        if all_models_data:
            output_filename = f"results/{dataset_name}_seq{seq_len}_pre{pre_len}_all_models_mean_mse.csv"
            output_df = pd.DataFrame({'horizon': horizons})
            for model, mean_mse in all_models_data.items():
                output_df[model] = mean_mse
            output_df.to_csv(output_filename, index=False)
            print(f"Saved all models mean MSE data to: {output_filename}")
        
        plt.xlabel('horizon')
        plt.ylabel('MSE')
        plt.xticks(horizons)
        plt.ylim(0.000, 0.025)  # 设置纵轴范围
        plt.title(f'MSE with 95% Confidence Interval for Each Model (seq{seq_len}_pre{pre_len})')
        plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
        plt.tight_layout(rect=[0, 0, 0.8, 1])
        
        # 保存图片文件
        plt.savefig(f'output_seq{seq_len}_pre{pre_len}.png', bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        print(f"Saved plot to: output_seq{seq_len}_pre{pre_len}.png")

print("\n=== All combinations processed ===")