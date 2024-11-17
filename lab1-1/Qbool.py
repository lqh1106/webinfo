import json
import re
import timeit
from sympy import symbols
from sympy.logic.boolalg import to_cnf

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

def extract_variables(query_string):
    # 使用正则表达式提取变量
    pattern = r'\b(?!AND|OR|NOT)(\w+)\b'  # 匹配字母组成的单词
    found_variables = set(re.findall(pattern, query_string))
    return list(found_variables)

def parse_query(query_string, variables, synonyms_dict):
    # 将字符串中的关键字替换为相应的符号
    for var in variables:
        new_var = replace_with_center_word(var, synonyms_dict)
        query_string = query_string.replace(var, f'symbols("{new_var}")')

    # 将其他逻辑运算符替换为 SymPy 的语法
    query_string = query_string.replace('AND', '&').replace('OR', '|').replace('NOT', '~')

    # 使用 eval 将字符串转换为布尔表达式
    return eval(query_string)

def replace_operators(expr):
    # 将合取正常型中的符号替换为 AND, OR, NOT
    expr_str = str(expr)
    expr_str = expr_str.replace('&', 'AND')
    expr_str = expr_str.replace('|', 'OR')
    expr_str = expr_str.replace('~', 'NOT ')
    return expr_str

def process_boolean_query(input_query, synonyms_dict):
    variables = extract_variables(input_query)
    # print(f"提取的布尔变量: {variables}")
    symbols_dict = {var: symbols(var) for var in variables}
    query = parse_query(input_query, variables, synonyms_dict)
    # cnf_query = to_cnf(query, simplify=True)
    # cnf_query_str = replace_operators(cnf_query)
    # print(f"原始查询: {query}")
    # print(f"合取正常型: {cnf_query_str}")
    return query

def extract_cnf_expressions(cnf_string):
    """ 从 CNF 字符串中提取表达式，按括号拆分成列表 """
    expressions = []
    current_expression = []
    stack = []

    for char in cnf_string:
        if char == '(':
            # 当遇到 '(' 时，将当前表达式压入栈中并重置 current_expression
            stack.append(current_expression)
            current_expression = []
        elif char == ')':
            # 当遇到 ')' 时，完成当前表达式并从栈中恢复
            if stack:
                if current_expression:
                    expressions.append(' '.join(current_expression).strip())
                current_expression = stack.pop()
        else:
            # 处理其他字符（包括 AND, OR, NOT 和操作数）
            current_expression.append(char)

    # 检查最后的表达式没有被完全添加
    if current_expression:
        expressions.append(' '.join(current_expression).strip())
    
    return expressions

def estimate_length(expr, inverted_index):
    """估计表达式的倒排表总长，考虑仅包含 OR 和 NOT 运算符的情况"""
    # 删除空格以简化处理
    expr = expr.replace(" ", "")
    total_length = 0

    # 处理表达式，找到所有 NOT 和 OR 部分
    or_parts = expr.split('OR')
    
    for part in or_parts:
        part = part.strip()  # 清理空格
        # 判断是否为 NOT 开头
        if part.startswith('NOT'):
            # 如果是 NOT 语句，则跳过不计
            continue
        else:
            # 对于其他部分（不带 NOT 的），增加其频率
            if part in inverted_index:
                total_length += inverted_index[part][0]  # 倒排表第二项的长度
    
    return total_length

# 布尔查询函数支持括号
def boolean_query_with_frequency(inverted_index, query, synonyms_dict):
    # 去除多余的空格
    query = query.strip()
    
    # 定义操作符
    operators = {'AND', 'OR', 'NOT'}
    
    # 定义运算符的优先级
    precedence = {'AND': 2, 'OR': 1, 'NOT': 3}  
    tokens = query.replace('(', ' ( ').replace(')', ' ) ').split()
    cnf_query = process_boolean_query(query, synonyms_dict)
    # sub_expressions = re.findall(r'$(.*?)$', str(cnf_query))
    

def boolean_query_basic(inverted_index, query, synonyms_dict):
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
            if operator == 'AND':
                result_set = left[1].intersection(right[1])
                output_stack.append((len(result_set), result_set))
            elif operator == 'OR':
                result_set = left[1].union(right[1])
                output_stack.append((len(result_set), result_set))

    # 返回最终结果
    return output_stack[0][1] if output_stack else set()

def boolean_query_with_frequency_optimized(inverted_index, query, synonyms_dict):
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
            # 处理非操作符的标签，获取其对应的书籍 ID
            token = replace_with_center_word(token, synonyms_dict)
            data = inverted_index.get(token, [0])
            if len(data) > 1:
                output_stack.append((len(data[1]), set(data[1])))  # 存储频率和集合
            else:
                output_stack.append((0, set()))  # 如果没有 ID，入空集合
        elif token == '(':
            operator_stack.append(token)  # 左括号入栈
        elif token == ')':
            # 处理并弹出操作符直到遇到左括号
            while operator_stack and operator_stack[-1] != '(':
                operator = operator_stack.pop()
                right = output_stack.pop()
                if operator == 'NOT':
                    operand = right[1]  # 获取右操作数的集合
                    # 直接从倒排表中获取 A 的 ID 列表
                    not_term = replace_with_center_word(left[1], synonyms_dict)  # 获取 NOT 操作符前的词（假设在左侧）
                    not_data = inverted_index.get(not_term, [0])  # 获取 A 的倒排表
                    if len(not_data) > 1:  # 确保有 ID
                        all_ids = set(not_data[1])  # 只获取 A 的 ID
                    else:
                        all_ids = set()  # 没有 ID 的话，使用空集合
                    output_stack.append((len(all_ids - operand), all_ids - operand))
                else:  # AND 或 OR 操作
                    left = output_stack.pop()
                    combined_frequency = left[0] + right[0]  # 合并频率
                    if operator == 'AND':
                        result_set = left[1].intersection(right[1])  # 交集
                    elif operator == 'OR':
                        result_set = left[1].union(right[1])  # 并集
                    output_stack.append((len(result_set), result_set))

            operator_stack.pop()  # 弹出 '('
        else:
            # 处理当前操作符的优先级
            while (operator_stack and operator_stack[-1] != '(' and 
                   precedence[token] <= precedence[operator_stack[-1]]):
                operator = operator_stack.pop()
                right = output_stack.pop()
                if operator == 'NOT':
                    operand = right[1]  # 获取右操作数的集合
                    # 直接从倒排表中获取 A 的 ID 列表
                    not_term = replace_with_center_word(left[1], synonyms_dict)  # 获取 NOT 操作符前的词（假设在左侧）
                    not_data = inverted_index.get(not_term, [0])  # 获取 A 的倒排表
                    if len(not_data) > 1:  # 确保有 ID
                        all_ids = set(not_data[1])  # 只获取 A 的 ID
                    else:
                        all_ids = set()  # 没有 ID 的话，使用空集合
                    output_stack.append((len(all_ids - operand), all_ids - operand))
                else:  # AND 或 OR 操作
                    left = output_stack.pop()
                    combined_frequency = left[0] + right[0]  # 合并频率
                    if operator == 'AND':
                        result_set = left[1].intersection(right[1])  # 交集
                    elif operator == 'OR':
                        result_set = left[1].union(right[1])  # 并集
                    output_stack.append((len(result_set), result_set))
            operator_stack.append(token)

    # 处理剩余的运算符
    while operator_stack:
        operator = operator_stack.pop()
        right = output_stack.pop()
        if operator == 'NOT':
            operand = right[1]
            # 获取 NOT 操作数的频率和 ID 列表
            all_ids = set()
            for value in inverted_index.values():
                if len(value) > 1:
                    all_ids.update(set(value[1]))
            output_stack.append((len(all_ids - operand), all_ids - operand))
        else:  # AND 或 OR 操作
            left = output_stack.pop()
            combined_frequency = left[0] + right[0]  # 合并频率
            if operator == 'AND':
                result_set = left[1].intersection(right[1])  # 交集
            elif operator == 'OR':
                result_set = left[1].union(right[1])  # 并集
            output_stack.append((len(result_set), result_set))

    # 返回最终结果
    return output_stack[0][1] if output_stack else set()



if __name__ == "__main__":
    file_path = 'lab1-1/dataset/inverted_index.json'  
    synonyms_dict_path = 'lab1-1/dataset/dict_synonym.txt'  
    inverted_index = load_inverted_index(file_path)
    synonyms_dict = load_synonym_dict(synonyms_dict_path)

    # 示例查询

    queries = [
        "(动作 AND 剧情) OR (科幻 AND NOT 恐怖)",
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
            number=10
        )
        result = boolean_query_basic(inverted_index, query, synonyms_dict)  # 获取最后的结果用于输出
        result = sorted(result)
        print(f"查询语句：{query}")
        print(f"查询结果：{result}")
        print(f"查询耗时：{execution_time:.6f}秒")
