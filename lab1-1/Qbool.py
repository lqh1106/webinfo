import re

class InvertedIndex:
    def __init__(self):
        self.index = {}

    def add(self, word, doc_id):
        if word not in self.index:
            self.index[word] = set()
        self.index[word].add(doc_id)

    def get(self, word):
        return self.index.get(word, set())

    def get_all_documents(self):
        all_docs = set()
        for docs in self.index.values():
            all_docs.update(docs)
        return all_docs

def evaluate_query(query, index):
    def parse(tokens):
        stack = []
        output = []
        precedence = {'OR': 1, 'AND': 2, 'NOT': 3}

        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                stack.pop()  # 去掉 '('
            elif token in precedence:
                while (stack and stack[-1] in precedence and 
                       precedence[token] <= precedence[stack[-1]]):
                    output.append(stack.pop())
                stack.append(token)
            else:
                output.append(token)

        while stack:
            output.append(stack.pop())
        
        return output

    def eval_postfix(postfix):
        stack = []
        for token in postfix:
            if token in ["AND", "OR", "NOT"]:
                if token == "AND":
                    right = stack.pop()
                    left = stack.pop()
                    result = left.intersection(right)
                    stack.append(result)
                elif token == "OR":
                    right = stack.pop()
                    left = stack.pop()
                    result = left.union(right)
                    stack.append(result)
                elif token == "NOT":
                    operand = stack.pop()
                    all_docs = index.get_all_documents()  # 获取所有文档 ID
                    result = all_docs - operand  # 从所有文档中去掉包含该单词的文档
                    stack.append(result)
            else:
                stack.append(index.get(token))

        return stack.pop()

    # 分割和清理查询字符串
    tokens = re.findall(r'\w+|[()&|!]', query.replace('and', 'AND').replace('or', 'OR').replace('not', 'NOT'))
    postfix = parse(tokens)
    return eval_postfix(postfix)

# 示例倒排索引初始化
inverted_index = InvertedIndex()
inverted_index.add("动作", 1)
inverted_index.add("动作", 2)
inverted_index.add("剧情", 1)
inverted_index.add("科幻", 3)
inverted_index.add("恐怖", 4)

# 执行查询示例
query_1 = "(动作 OR 剧情) AND (科幻 OR NOT 恐怖)"
result = evaluate_query(query_1, inverted_index)

print("查询结果:", result)