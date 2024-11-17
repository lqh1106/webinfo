import re
from sympy import symbols
from sympy.logic.boolalg import to_cnf

def extract_variables(query_string):
    # 使用正则表达式提取变量
    pattern = r'\b(?!AND|OR|NOT)(\w+)\b'  # 匹配字母组成的单词
    found_variables = set(re.findall(pattern, query_string))
    return list(found_variables)

def parse_query(query_string, variables):
    # 将字符串中的关键字替换为相应的符号
    for var in variables:
        query_string = query_string.replace(var, f'symbols("{var}")')

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

def process_boolean_query(input_query):
    variables = extract_variables(input_query)
    print(f"提取的布尔变量: {variables}")
    symbols_dict = {var: symbols(var) for var in variables}
    query = parse_query(input_query, variables)
    cnf_query = to_cnf(query, simplify=True)
    cnf_query_str = replace_operators(cnf_query)
    print(f"原始查询: {query}")
    print(f"合取正常型: {cnf_query_str}")
    return cnf_query_str

# 输入布尔查询
input_query = "((人工智能 AND NOT (机器人 OR 自动化)) OR (大数据 AND (分析 OR 可视化))) AND (技术 OR 创新)"

# 自动提取布尔变量
variables = extract_variables(input_query)
print(f"提取的布尔变量: {variables}")

# 定义布尔变量
symbols_dict = {var: symbols(var) for var in variables}

# 解析用户输入的查询
query = parse_query(input_query, variables)

# 将布尔查询转换为合取正常型
cnf_query = to_cnf(query, simplify=True)
cnf_query_str = replace_operators(cnf_query)

# 输出结果

print(f"合取正常型: {cnf_query_str}")



