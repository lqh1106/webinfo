import json
import struct
import pickle

def block_storage(ids, block_size=5):
    """
    按块存储方式对文档ID列表进行压缩。
    
    :param ids: 原始文档ID列表
    :param block_size: 块的大小
    :return: 压缩后的文档ID列表
    """
    blocks = []
    for i in range(0, len(ids), block_size):
        block = ids[i:i + block_size]
        blocks.append(block)
    return blocks

def front_encoding(ids):
    """
    对文档ID列表进行差分编码。
    
    :param ids: 原始文档ID列表
    :return: 压缩后的文档ID列表
    """
    if not ids:
        return []
    # 使用差分编码
    encoded = [ids[0]]  # 加入第一个ID
    for i in range(1, len(ids)):
        encoded.append(ids[i] - ids[i - 1])
    return encoded

def front_encoding_index(inverted_index):
    """
    对倒排索引进行压缩。
    
    :param inverted_index: 原始倒排索引
    :return: 压缩后的倒排索引
    """
    compressed_index = {}
    
    for tag, (frequency, ids) in inverted_index.items():
        # 计算间距
        gaps = front_encoding(ids)

        compressed_index[tag] = [frequency, gaps]        
    
    return compressed_index

def compress_inverted_index(inverted_index):
    """
    对倒排索引进行压缩。
    
    :param inverted_index: 原始倒排索引
    :return: 压缩后的倒排索引
    """
    compressed_index = {}
    
    for tag, (frequency, ids) in inverted_index.items():
        # 计算间距
        gaps = front_encoding(ids)
        
        # 使用可变长度编码进行压缩
        encoded_gaps = variable_length_encode(gaps)
        compressed_index[tag] = [frequency, encoded_gaps]        
    
    return compressed_index

def variable_length_encode(gaps):
    """
    对间距进行可变长度编码。
    
    :param gaps: 要编码的间距列表
    :return: 编码后的字节流
    """
    byte_array = bytearray()
    
    for g in gaps:
        if g < 128:
            # G < 128
            byte_array.append(g | 0x80)  # 1位延续位为1
        else:
            # G >= 128
            while g >= 128:
                byte_array.append(g & 0x7F)  # 低7位
                g >>= 7  # 右移7位
            # print(g | 0x80)
            byte_array.append(g | 0x80)  # 最后的字节
            
    return bytes(byte_array)

def variable_length_decode(encoded_bytes):
    """
    对编码后的字节流进行解码。
    
    :param encoded_bytes: 编码后的字节流
    :return: 解码后的间距列表
    """
    gaps = []
    i = 0
    current_gap = 0
    k = 0
    while i < len(encoded_bytes):
        byte = encoded_bytes[i]
        current_gap += (byte & 0x7F) << (7 * k)  # 低7位
        if(byte & 0x80 == 0):
            i += 1
            k += 1
            continue
        else:
            gaps.append(current_gap)
            current_gap = 0
            k = 0
        i += 1
    return gaps

def decompress_inverted_index(compressed_index):
    """
    解压缩倒排索引。
    
    :param compressed_index: 压缩后的倒排索引
    :return: 原始倒排索引
    """
    inverted_index = {}
    
    for tag, (frequency, encoded_gaps) in compressed_index.items():
        gaps = variable_length_decode(encoded_gaps)
        ids = [gaps[0]]  # 第一个间距直接作为id
        
        # 根据间距重建ID列表
        for gap in gaps[1:]:
            ids.append(ids[-1] + gap)
        
        inverted_index[tag] = [frequency, ids]
    
    return inverted_index


# 存储倒排索引
def save_index(compressed_index, filename):
    with open(filename, 'wb') as file:
        pickle.dump(compressed_index, file)

# 读取倒排索引
def load_index(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 使用示例
if __name__ == '__main__':
    # 假设这里从一个JSON文件加载原始倒排索引
    with open('lab1-1/dataset/book_index.json', 'r', encoding='utf-8') as f:
        inverted_index = json.load(f)

    save_index(inverted_index, 'lab1-1/dataset/index.pkl')

    front_encoded_index = front_encoding_index(inverted_index)

    save_index(front_encoded_index, 'lab1-1/dataset/front_encoded_index.pkl')
    
    # 压缩倒排索引
    compressed_index = compress_inverted_index(inverted_index)

    # 存储压缩后的倒排索引到文件
    save_index(compressed_index, 'lab1-1/dataset/compressed_index.pkl')

    # 从文件中读取压缩后的倒排索引
    loaded_index = load_index('lab1-1/dataset/compressed_index.pkl')

    # 解压缩倒排索引
    decompressed_index = decompress_inverted_index(loaded_index)

    # 将压缩后的倒排索引写入到一个新的JSON文件中
    with open('lab1-1/dataset/front_encoded_index.json', 'w', encoding='utf-8') as f:
        json.dump(decompressed_index, f, ensure_ascii=False, indent=4)
