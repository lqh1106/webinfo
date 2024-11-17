import json
import re
import time
import timeit

# 读取倒排索引
def load_inverted_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_synonym_dict(file_path):
    synonym_dict = {}
    code = ''
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 使用正则表达式匹配 '=', '#' 或 '@' 作为分隔符
            newcode, synonyms = re.split(r'[=#@]\s', line.strip(), maxsplit=1)
            synonym_list = synonyms.split(' ')
            if synonym_list:  # 确保有近义词
                if newcode[:7] != code[0:7]:
                    code = newcode
                    center_word = synonym_list[0]  # 第一个词作为中心词
                for word in synonym_list:
                    synonym_dict[word] = center_word
    return synonym_dict

def replace_with_center_word(word, synonym_dict):
    new_word = synonym_dict.get(word, word)
    return new_word

# 布尔查询函数支持括号
def boolean_query(inverted_index, query, synonyms_dict):
    # 去除多余的空格
    query = query.strip()
    
    # 定义操作符
    operators = {'AND', 'OR', 'NOT'}
    
    # 定义运算符的优先级
    precedence = {'AND': 2, 'OR': 1, 'NOT': 3}
    
    # 用于存储操作数和运算符的栈
    output_stack = []
    operator_stack = []

    # 处理查询
    tokens = query.replace('(', ' ( ').replace(')', ' ) ').split()
    
    for token in tokens:
        if token not in operators and token != '(' and token != ')':
            # 不在运算符中的是标签，获取其对应书籍ID
            token = replace_with_center_word(token, synonyms_dict)
            data = inverted_index.get(token, [0])
            if len(data) > 1:
                output_stack.append(set(data[1]))
            else:
                output_stack.append(set())
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                operator = operator_stack.pop()
                if operator == 'NOT':
                    operand = output_stack.pop()
                    all_ids = set()
                    for ids in inverted_index.values():
                        all_ids.update(ids)
                    output_stack.append(all_ids - operand)
                else:
                    right = output_stack.pop()
                    left = output_stack.pop()
                    if operator == 'AND':
                        output_stack.append(left.intersection(right))
                    elif operator == 'OR':
                        output_stack.append(left.union(right))
            operator_stack.pop()  # 弹出 '('
        else:
            while (operator_stack and operator_stack[-1] != '(' and 
                   precedence[operator] <= precedence[operator_stack[-1]]):
                operator = operator_stack.pop()
                if operator == 'NOT':
                    operand = output_stack.pop()
                    all_ids = set()
                    for ids in inverted_index.values():
                        all_ids.update(ids)
                    output_stack.append(all_ids - operand)
                else:
                    right = output_stack.pop()
                    left = output_stack.pop()
                    if operator == 'AND':
                        output_stack.append(left.intersection(right))
                    elif operator == 'OR':
                        output_stack.append(left.union(right))
            operator_stack.append(token)

    # 处理剩余的运算符
    while operator_stack:
        operator = operator_stack.pop()
        if operator == 'NOT':
            operand = output_stack.pop()
            all_ids = set()
            for ids in inverted_index.values():
                all_ids.update(ids)
            output_stack.append(all_ids - operand)
        else:
            right = output_stack.pop()
            left = output_stack.pop()
            if operator == 'AND':
                output_stack.append(left.intersection(right))
            elif operator == 'OR':
                output_stack.append(left.union(right))

    # 返回最终结果
    return output_stack[0] if output_stack else set()

def boolean_query_with_frequency(inverted_index, query, synonyms_dict):
    # 去除多余的空格
    query = query.strip()
    
    # 定义操作符
    operators = {'AND', 'OR', 'NOT'}
    
    # 定义运算符的优先级
    precedence = {'AND': 2, 'OR': 1, 'NOT': 3}
    
    # 用于存储操作数和运算符的栈
    output_stack = []
    operator_stack = []

    # 处理查询
    tokens = query.replace('(', ' ( ').replace(')', ' ) ').split()
    
    for token in tokens:
        if token not in operators and token != '(' and token != ')':
            # 不在运算符中的是标签，获取其对应书籍ID
            token = replace_with_center_word(token, synonyms_dict)
            data = inverted_index.get(token, [0])
            if len(data) > 1:
                output_stack.append((len(data[1]), set(data[1])))  # 存储频率和集合
            else:
                output_stack.append((0, set()))  # 如果没有书籍ID，加入频率为0的空集合
        elif token == '(':
            operator_stack.append(token)
        elif token == ')':
            while operator_stack and operator_stack[-1] != '(':
                operator = operator_stack.pop()
                if operator == 'NOT':
                    operand = output_stack.pop()[1]  # 只取集合部分
                    all_ids = set()
                    for value in inverted_index.values():
                        if len(value) > 1:
                            all_ids.update(set(value[1]))
                    output_stack.append((len(all_ids - operand), all_ids - operand))
                else:
                    right = output_stack.pop()
                    left = output_stack.pop()
                    combined_frequency = left[0] + right[0]  # 合并频率
                    if operator == 'AND':
                        result_set = left[1].intersection(right[1])
                        output_stack.append((len(result_set), result_set))
                    elif operator == 'OR':
                        result_set = left[1].union(right[1])
                        output_stack.append((len(result_set), result_set))
            operator_stack.pop()  # 弹出 '('
        else:
            while (operator_stack and operator_stack[-1] != '(' and 
                   precedence[token] <= precedence[operator_stack[-1]]):
                operator = operator_stack.pop()
                if operator == 'NOT':
                    operand = output_stack.pop()[1]  # 只取集合部分
                    all_ids = set()
                    for value in inverted_index.values():
                        if len(value) > 1:
                            all_ids.update(set(value[1]))
                    output_stack.append((len(all_ids - operand), all_ids - operand))
                else:
                    right = output_stack.pop()
                    left = output_stack.pop()
                    combined_frequency = left[0] + right[0]  # 合并频率
                    if operator == 'AND':
                        result_set = left[1].intersection(right[1])
                        output_stack.append((len(result_set), result_set))
                    elif operator == 'OR':
                        # 估计 OR 操作的大小并在这里可能进行优化
                        result_set = left[1].union(right[1])
                        output_stack.append((len(result_set), result_set))
            operator_stack.append(token)

    # 处理剩余的运算符
    while operator_stack:
        operator = operator_stack.pop()
        if operator == 'NOT':
            operand = output_stack.pop()[1]
            all_ids = set()
            for id_list in inverted_index.values():
                all_ids.update(set(id_list))
            output_stack.append((len(all_ids - operand), all_ids - operand))
        else:
            right = output_stack.pop()
            left = output_stack.pop()
            combined_frequency = left[0] + right[0]  # 合并频率
            if operator == 'AND':
                result_set = left[1].intersection(right[1])
                output_stack.append((len(result_set), result_set))
            elif operator == 'OR':
                result_set = left[1].union(right[1])
                output_stack.append((len(result_set), result_set))

    # 返回最终结果
    return output_stack[0][1] if output_stack else set()

def test_function():
    result = sum(range(1000))
# 示例使用
if __name__ == "__main__":
    file_path = 'lab1-1/dataset/inverted_index.json'  # 根据实际情况设置文件路径
    synonyms_dict_path = 'lab1-1/dataset/dict_synonym.txt'  # 根据实际情况设置文件路径
    inverted_index = load_inverted_index(file_path)
    synonyms_dict = load_synonym_dict(synonyms_dict_path)

    # 示例查询

    queries = [
        "(鳞次栉比)",
        "(苹果 AND 橙子) OR (香蕉 AND NOT 葡萄)",
        "((山脉 OR 海洋) AND (夏天 AND NOT 雨天)) OR (城市 AND (夜景 OR 商业))",
        "((学习 AND (编程 OR 数据分析)) AND (工具 OR 资源))",
        "((疫情 OR 疫苗) AND NOT (恐慌 OR 死亡)) OR (健康 AND (生活方式 OR 饮食))",
        "((人工智能 AND NOT (机器人 OR 自动化)) OR (大数据 AND (分析 OR 可视化))) AND (技术 OR 创新)"
    ]
# 进行每个查询的时间测量
    for query in queries:
        execution_time = timeit.timeit(
            lambda: boolean_query_with_frequency(inverted_index, query, synonyms_dict),
            number=100
        )
        result = boolean_query_with_frequency(inverted_index, query, synonyms_dict)  # 获取最后的结果用于输出
        result = sorted(result)
        print(f"查询语句：{query}")
        print(f"查询结果：{result}")
        print(f"查询耗时：{execution_time:.6f}秒")
