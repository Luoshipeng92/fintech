import os
import json
import argparse
import numpy as np

from tqdm import tqdm
import jieba  # 用於中文文本分詞
import pdfplumber  # 用於從PDF文件中提取文字的工具
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
import torch
from torch.nn.functional import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 文字切塊(chunking)
def split_text(text, key, chunk_size=400, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    
    return text_splitter.create_documents([text],metadatas=[{"source": key}])

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {int(file.replace('.pdf', '')): read_pdf(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
    pdf_text = ''
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁      
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    clean_pdf_text = pdf_text.replace('\n', '')
    pdf.close()  # 關閉PDF文件
    chunks = split_text(clean_pdf_text)

    return chunks  # 返回萃取出的文本


# 初始化 Sentence-BERT 模型
# model = SentenceTransformer('shibing624/text2vec-base-chinese')  # 或其他 SBERT 模型
# model = SentenceTransformer('intfloat/multilingual-e5-large')  # 或其他 SBERT 模型
model = SentenceTransformer('BAAI/bge-m3')  # 或其他 SBERT 模型
def bert_retrieve(qs, source, corpus_dict, n):
    """
    使用 Sentence-BERT 來檢索與查詢最相關的文檔。
    
    參數：
    - qs (str): 查詢語句
    - source (list): 需要檢索的文檔索引列表
    - corpus_dict (dict): 文檔集合，格式為 {文件名: 文檔內容}
    - n (int): 返回最相關的文檔數量

    返回：
    - List[str]: 最相關的文檔名稱列表
    """
    
    # 過濾文檔集合，只保留在 source 中的文檔
    # filtered_corpus = {key: corpus_dict[key] for key in source if key in corpus_dict}
    chunks = []
    for file in source:
        chunk_file = split_text(corpus_dict[int(file)], file)
        chunks.extend(chunk_file)
    # print(chunks)
    # 嵌入查詢和過濾後的文檔
    
    query_embedding = model.encode(qs, convert_to_tensor=True)
    # corpus_embeddings = model.encode(list(filtered_corpus.values()), convert_to_tensor=True)
    chunk_texts = [chunk.page_content for chunk in chunks]  # 獲取每個 chunk 的文字內容
    chunk_embeddings = model.encode(chunk_texts, convert_to_tensor=True)
    # 計算查詢與每篇文檔的餘弦相似度
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    
    # 找到相似度最高的 n 個文檔索引
    top_results = torch.topk(similarities, k=n)

    # 將結果與文件名對應
    # res = [list(filtered_corpus.keys())[idx] for idx in top_results.indices]
    res = [chunks[idx].metadata["source"] for idx in top_results.indices]
    # 輸出結果
    print(res)
    return res

# 根據查詢語句和指定的來源，檢索答案
def BM25_retrieve(qs, source, corpus_dict, n):
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # [TODO] 可自行替換其他檢索方式，以提升效能
    
    tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n)  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    if n == 1 and ans:
        a = ans[0]
        # 找回與最佳匹配文本相對應的檔案名
        res = [key for key, value in corpus_dict.items() if value == a]
        print(res)
        return res[0]  # 回傳檔案名
    else:
        res = [key for key, value in corpus_dict.items() if value in ans]
        print(res)
        return res[:n]

    # 找回與最佳匹配文本相對應的檔案名



if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, required=True, help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, required=True, help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數

    answer_dict = {"answers": []}  # 初始化字典

    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    with open(os.path.join(args.source_path, 'faq/updated_pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict1 = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict1 = {int(key): value for key, value in key_to_source_dict1.items()}
        
    with open(os.path.join(args.source_path, 'finance/finance_OCRRR.json'), 'rb') as e_s:
        key_to_source_dict2 = json.load(e_s)  # 讀取參考資料文件
        key_to_source_dict2 = {int(key): value for key, value in key_to_source_dict2.items()}
        
    with open(os.path.join(args.source_path, 'insurance/insurance.json'), 'rb') as i_s:
        key_to_source_dict3 = json.load(i_s)  # 讀取參考資料文件
        key_to_source_dict3 = {int(key): value for key, value in key_to_source_dict3.items()}        
    

    for q_dict in qs_ref['questions']:
        if q_dict['category'] == 'finance':
            corpus_dict_finance = {key: str(value) for key, value in key_to_source_dict2.items() if key in q_dict['source']}
            # 進行檢索
            # retrieved_1 = bert_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance,1)
            retrieved = 0
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            corpus_dict_insurance = {key: str(value) for key, value in key_to_source_dict3.items() if key in q_dict['source']}
            # retrieved_1 = bert_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance,1)
            retrieved = 0
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict1.items() if key in q_dict['source']}
            retrieved = bert_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq,1)
            # retrieved = 0
            # 將結果加入字典
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved[0]})
        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    # 將答案字典保存為json文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符..