import pandas as pd


global_dataframe = pd.DataFrame(columns=["qid", "source", "score"])

"""results裡的score表示方法要改
   函式要多傳入一個qid
"""
def make_dataframe(qid, source_list, scores):
    global global_dataframe

    # 確保分數是 numpy array 或可計算的格式
    scores = scores.numpy() if hasattr(scores, 'numpy') else scores

    min_score = min(scores)
    max_score = max(scores)

    for i in range(len(source_list)):
        source = source_list[i]
        score = scores[i]  # 使用每個分數
        normalized_score = (score - min_score) / (max_score - min_score)

        # 組成新的 DataFrame 行
        new_row = pd.DataFrame([{
            "qid": qid,
            "source": source,
            "score": normalized_score
        }])

        # 合併到全域的 DataFrame
        global_dataframe = pd.concat([global_dataframe, new_row], ignore_index=True)

# 使用這個函式時，需要傳入 qid、model_name、source_list 和 results
"""file_name設成模型名稱"""
def output_dataframe(file_name):
    global global_dataframe
    global_dataframe.to_csv(file_name, index=False, encoding='utf-8-sig')
    print(f"CSV file '{file_name}' has been created.")
