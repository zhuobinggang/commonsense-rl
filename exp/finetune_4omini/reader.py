import os
import random

def read_lines_from_json_files_in_directory(path):
    """
    读取指定文件夹内所有 JSON 文件的内容，逐行读取，舍弃空行，返回一个字符串列表。

    Args:
        path (str): 文件夹路径

    Returns:
        list: 包含非空行的字符串列表
    """
    if not os.path.isdir(path):
        raise ValueError(f"提供的路径 '{path}' 不是一个有效的文件夹。")

    result = []

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(path):
        if file_name.endswith('.json'):  # 只处理 JSON 文件
            file_path = os.path.join(path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()  # 去掉前后空白字符
                        if line:  # 跳过空行
                            result.append(line)
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")

    return result


def lines_random_then_output(lines, output_file='out.json'):
    """
    将输入列表的顺序随机打乱并保存到文件中，每行保存一个字符串。

    Args:
        lines (list): 要随机打乱的字符串列表。
        output_file (str): 输出文件路径，默认为 'out.json'。

    Returns:
        None
    """
    if not isinstance(lines, list):
        raise ValueError("输入必须是一个列表。")
    
    # 打乱列表顺序
    random.shuffle(lines)
    
    try:
        # 将打乱后的内容写入文件
        with open(output_file, 'w', encoding='utf-8') as file:
            for line in lines:
                file.write(line + '\n')  # 每行写入一个字符串，后接换行符
        print(f"打乱后的内容已保存到文件: {output_file}")
    except Exception as e:
        print(f"写入文件 {output_file} 时发生错误: {e}")

# 示例使用
# lines = ["line1", "line2", "line3", "line4"]
# lines_random_then_output(lines)


def lines_random_then_output_v2(lines, output_file='out.jsonl'):
    import json
    """
    将输入的 Python 字典格式列表随机打乱，并以 JSON Lines 格式输出到文件。

    Args:
        lines (list): 要处理的字典格式字符串列表。
        output_file (str): 输出文件路径，默认为 'out.jsonl'。

    Returns:
        None
    """
    if not isinstance(lines, list):
        raise ValueError("输入必须是一个列表。")

    # 打乱列表顺序
    random.shuffle(lines)

    try:
        # 将打乱后的内容以 JSON Lines 格式写入文件
        with open(output_file, 'w', encoding='utf-8') as file:
            for line in lines:
                # 将每个字典对象转换为 JSON 格式字符串并写入文件
                json_object = eval(line)  # 将字符串解析为 Python 字典
                json_line = json.dumps(json_object, ensure_ascii=False)  # 转为标准 JSON
                file.write(json_line + '\n')  # 每行一个 JSON 对象
        print(f"打乱后的内容已保存到文件: {output_file}")
    except Exception as e:
        print(f"写入文件 {output_file} 时发生错误: {e}")
