import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import json
import os
from coat_utils import *
from collections import defaultdict
from config.coat_pattern import start_pattern, agent_pattern, user_pattern
from config.thanks_pattern import coat_agent, coat_user
from config.recommend_pattern import coat_rec
from coat_attr import generate_gender_dialogue, generate_jacket_dialogue, generate_color_dialogue

def load_data():
    # 读取数据，不过里面只有（userid，itemid，gender，age，location，rating）
    coat = pd.read_csv("./data/coat/coat_info.csv")
    # 这里把age做了一次map的转换，不知道为啥
    coat['age'] = coat['age'].apply(lambda age: age_map(age))
    # 用于从文本文件中加载数据并将其转换为 NumPy 数组，dtype：数据类型
    item_feature = np.genfromtxt('./data/coat/item_features.ascii', dtype=None)

    return (coat, item_feature)

def calculate_user_preference(data):
    # 这个数据的2个维 (coat, item_feature)
    # 衣服的8条数据
    coat = data[0]
    # 商品特征
    item_feature = data[1]
    # 指定了 set 作为默认工厂函数。这意味着，如果你尝试访问字典中不存在的键，defaultdict 将自动为该键创建一个新的空 set 对象作为默认值，并返回这个空 set
    user_record = defaultdict(set)
    #  NumPy 中用于创建一个形状为 (300, 3) 的300 行和 3 列数组，其中所有元素都初始化为 0
    item_matrix = np.zeros((300, 3))
    # row属性，（userid，itemid，gender，age，location，rating）
    for _, row in coat.iterrows():
        # 用户id
        user_id = int(row['user'])
        # 商品id
        item_id = int(row['item'])
        # 用户id，都加商品id
        user_record[user_id].add(item_id)
        # 通过商品id，获取出这3个属性
        gender, jacket, color = get_item_index(item_feature, item_id)
        # 放到商品行里面（3列=就是这个商品的属性）
        item_matrix[item_id][0] = gender
        item_matrix[item_id][1] = jacket
        item_matrix[item_id][2] = color
    # DataFrame 是一个二维的、表格型的数据结构
    item_csv = pd.DataFrame(item_matrix)
    #
    item_csv.insert(item_csv.shape[1], 'label', 0)
    # 定义3个属性
    item_csv.columns = ['gender','jacket', 'color', 'label']
    #
    features_cols = ['gender', 'jacket', 'color']
    user_preference = defaultdict(list)
    # 遍历300个用户
    for user in range(300):
        user_item_csv = item_csv.copy()
        record = list(user_record[user])
        for item in record:
            user_item_csv.loc[item, 'label'] = 1
        X = user_item_csv[features_cols]
        Y = user_item_csv.label

        cls = lgb.LGBMClassifier(importance_type='gain')
        cls.fit(X, Y)

        indices = np.argsort(cls.booster_.feature_importance(importance_type='gain'))
        feature = [features_cols[i] for i in indices]

        user_preference[user] = feature
    
    return  user_preference

def get_user_item_info(data):
    user_info = {}
    item_info = {}
    coat = data[0]
    item_feature = data[1]
    for _, row in coat.iterrows():
        user_id = int(row['user'])
        item_id = int(row['item'])
        user_info[user_id] = {}
        user_info[user_id]['age'] = get_user_age(row['age'])
        user_info[user_id]['gender'] = get_user_gender(row['gender'])

        if item_id not in item_info.keys():
            gender, jacket, color = get_item_index(item_feature, item_id)
            item_info[item_id] = {}
            item_info[item_id]['gender'] = get_item_gender(gender)
            item_info[item_id]['jacket'] = get_item_type(jacket)
            item_info[item_id]['color'] = get_item_color(color)
    
    return  user_info, item_info

def calculate_attr_weights(data):
    gender_all = {}
    jacket_all = {}
    color_all = {}
    coat = data[0]
    item_feature = data[1]
    for _, row in coat.iterrows():
        item_id = int(row['item'])
        gender, jacket, color = get_item_index(item_feature, item_id)
        if gender not in gender_all.keys():
            gender_all[gender] = 1
        else:
            gender_all[gender] += 1
        
        if jacket not in jacket_all.keys():
            jacket_all[jacket] = 1
        else:
            jacket_all[jacket] += 1
        
        if color not in color_all.keys():
            color_all[color] = 1
        else:
            color_all[color] += 1
    
    gender_weight = [gender_all[i] for i in sorted(gender_all)]
    jacket_weight = [jacket_all[i] for i in sorted(jacket_all)]
    color_weight = [color_all[i] for i in sorted(color_all)]

    return (gender_weight, jacket_weight, color_weight), (gender_all, jacket_all, color_all)


if __name__ == '__main__':
    # 加载数据
    coat_data = load_data()
    #
    user_preference = calculate_user_preference(coat_data)
    user_info, item_info = get_user_item_info(coat_data)
    weights, attr_counts = calculate_attr_weights(coat_data)
    print("data load complete")
    dialogue_info = {}
    dialogue_id = 0
    for _, row in coat_data[0].iterrows():
        user_id = int(row['user'])
        item_id = int(row['item'])

        new_dialogue = {}
        new_dialogue["user_id"] = user_id
        new_dialogue["item_id"] = item_id

        new_dialogue["user_gender"] = user_info[user_id]["gender"]
        new_dialogue["user_age"] = user_info[user_id]["age"]
        new_dialogue["content"] = {}
        new_dialogue["content"]["start"] = random.choice(start_pattern)

        dialouge_order = user_preference[user_id]
        tmp_new_dialogue = []

        for slot in dialouge_order:
            if slot == "gender":
                gender_val = item_info[item_id]["gender"]
                utterance = generate_gender_dialogue(agent_pattern, user_pattern, gender_val, attr_counts[0], weights[0])
            elif slot == "jacket":
                jacket_val = item_info[item_id]["jacket"]
                utterance = generate_jacket_dialogue(agent_pattern, user_pattern, jacket_val, attr_counts[1], weights[1])
            elif slot == "color":
                color_val = item_info[item_id]["color"]
                utterance = generate_color_dialogue(agent_pattern, user_pattern, color_val, attr_counts[2], weights[2])
            tmp_new_dialogue.append(utterance)
        end_dialogue = {}
        end_dialogue["rec"] = random.choice(coat_rec)
        end_dialogue["thanks_user"] = random.choice(coat_user)
        end_dialogue["thanks_agent"] = random.choice(coat_agent)
        tmp_new_dialogue.append(end_dialogue)
        print("finish:", dialogue_id)
        start_index = 0
        end_index = 0
        step = 0
        name = ["Q1", "A1", "Q2", "A2", "Q3", "A3", "Q4", "A4", "Q5", "A5", "Q6", "A6", "Q7", "A7", "Q8", "A8", "Q9", "A9", "Q10", "A10", "Q11", "A11", "Q12", "A12"]
        for dia in tmp_new_dialogue:
            end_index = len(dia) + end_index
            tmp_name = name[start_index : end_index]
            tmp_dia = []
            for _, v in dia.items():
                tmp_dia.append(v)
            for i, val in enumerate(tmp_name):
                new_dialogue["content"][val] = tmp_dia[i]
            start_index = end_index
        dialogue_info[dialogue_id] = new_dialogue
        dialogue_id = dialogue_id + 1

    print("to the end...")
    res_path = './res/coat/'
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    with open(res_path + 'dialogue_info_coat.json', 'w') as f:
        json.dump(dialogue_info, f, indent=4)


            



