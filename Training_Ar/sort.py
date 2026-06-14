import pandas as pd
import glob
import os

# 定义列名
column_names = ['ID', 'X', 'rho', 'v1', 'Ttrans', 'p', 'p1', 'q1', 'mv1v1v1', 'mvvv1v1']

def process_file(file_path):
    # 获取文件所在的子文件夹名称
    subfolder_name = os.path.basename(os.path.dirname(file_path))
    
    # 读取文件头以提取 I 的值
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 提取 ITEM: NUMBER OF CELLS 的值
        i_value = int(lines[3].strip())  # 第 4 行保存了 I 的值

    # 跳过前 9 行，读取数据内容
    data = pd.read_csv(file_path, delim_whitespace=True, names=column_names, skiprows=9)
    
    # 按 ID 排序
    sorted_data = data.sort_values(by='ID', ascending=True)
    
    # 构建正激波输出文件名
    fwd_output_file = os.path.join(
        os.path.dirname(file_path),
        f"fwd_{os.path.basename(file_path).replace('tmp_flowvss.', '').split('.')[-1]}.dat"
    )
    
    # 写入 Tecplot 表头和正激波数据
    with open(fwd_output_file, 'w') as f:
        f.write("TITLE = \"DSMC\"\n")
        f.write("VARIABLES = \"ID\" \"X\" \"rho\" \"v1\" \"Ttrans\" \"p\" \"p1\" \"q1\" \"mv1v1v1\" \"mvvv1v1\"\n")
        f.write(f"ZONE T=\"{subfolder_name}\", I={i_value}, J=1, K=1, F=POINT\n")  # 使用子文件夹名称
        sorted_data.to_csv(f, sep=' ', index=False, header=False)
    
    print(f"Forward shock data from '{file_path}' has been sorted and saved to '{fwd_output_file}' with headers.")

    # 创建逆激波数据
    inverted_data = sorted_data.copy()
    inverted_data['X'] = -inverted_data['X']  # X 取负
    inverted_data['v1'] = -inverted_data['v1']  # v1 取负
    inverted_data['q1'] = -inverted_data['q1']  # q1 取负
    inverted_data['mv1v1v1'] = -inverted_data['mv1v1v1']  # mv1v1v1 取负
    # 其他变量保持不变

    # 按 X 排序，使其从小到大
    inverted_data = inverted_data.sort_values(by='X', ascending=True)

    # 构建逆激波输出文件名
    inv_output_file = os.path.join(
        os.path.dirname(file_path),
        f"inv_{os.path.basename(file_path).replace('tmp_flowvss.', '').split('.')[-1]}.dat"
    )

    # 写入 Tecplot 表头和逆激波数据
    with open(inv_output_file, 'w') as f:
        f.write("TITLE = \"DSMC\"\n")
        f.write("VARIABLES = \"ID\" \"X\" \"rho\" \"v1\" \"Ttrans\" \"p\" \"p1\" \"q1\" \"mv1v1v1\" \"mvvv1v1\"\n")
        f.write(f"ZONE T=\"{subfolder_name}_inv\", I={i_value}, J=1, K=1, F=POINT\n")
        inverted_data.to_csv(f, sep=' ', index=False, header=False)
    
    print(f"Inverted shock data from '{file_path}' has been saved to '{inv_output_file}'.")

# 使用 glob 递归获取所有子文件夹中的 tmp_flowvhs.* 文件的路径
file_paths = glob.glob('./**/tmp_flowvss.*', recursive=True)

# 检查是否有文件匹配
if file_paths:
    for file_path in file_paths:
        # 跳过文件名为 tmp_flowvss.0 的文件
        if file_path.endswith("tmp_flowvss.0"):
            print(f"Skipping file '{file_path}'")
            continue
        process_file(file_path)
else:
    print("No files found matching pattern './**/tmp_flowvss.*'.")
