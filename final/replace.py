import re
import json

# 讀取 JSON 檔案
with open("sorted_data1.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 定義函數來過濾內容，僅保留中文、逗號和句號
def filter_text(content):
    return re.sub(r'[^，。、\u4e00-\u9fa5]', '', content)

# 遍歷 JSON 結構，對每個值應用過濾
filtered_data = {key: filter_text(value) for key, value in data.items()}

# 將處理後的資料寫入新的 JSON 檔案
with open("sorted_data2.json", "w", encoding="utf-8") as file:
    json.dump(filtered_data, file, ensure_ascii=False, indent=4)