from QuIPU import QuIPU
import logging
import argparse
import os
import pandas as pd
import multiprocessing as mp
import numpy as np
from sentence_transformers import SentenceTransformer

mechanisms_embeddings = ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl']
mechanisms_sampling = ['CusText', 'SanText', 'TEM', 'WBB']
mechanisms_sota = ['AEA', 'FEA']

epsilons = [1, 5, 10, 12.5, 15, 17.5, 20, 25, 30, 50]
levels = [3, 4, 5]
winsize = [12, 14, 16]

model = SentenceTransformer('distilbert-base-uncased')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='QuIPU')
    parser.add_argument('--query_logs_embeddings', '-qle', type=str, help='Path to the query logs embeddings', default='./QuIPU/queryLogs/query_logs_embeddings.npz')
    parser.add_argument('--query_logs', '-ql', type=str, help='Path to the query logs', default='./QuIPU/queryLogs/query_logs.csv')
    parser.add_argument('--collection', '-c', type=str, help='Collection name', default='msmarco-dl19')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    #create a file handler for the logger
    #clear the log file
    open(f'./logs/logger_{args.collection}.log', 'w').close()
    handler = logging.FileHandler(f'./logs/logger_{args.collection}.log')
    #create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    #add the handlers to the logger
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Arguments:  Query logs embeddings: {args.query_logs_embeddings}, Collection: {args.collection}")
    #obfuscated_queries = pd.read_csv(f"data/obfuscation/{args.collection}/{args.mechanism}/obfuscatedText_{args.mechanism}_{eps}.csv")
    attack = QuIPU(logger = logger, query_logs = args.query_logs, query_logs_embeddings = args.query_logs_embeddings, model = model, collection = args.collection)
    for mech in mechanisms_sampling:
        if mech in mechanisms_sampling:
            for eps in epsilons:
                df = pd.read_csv(f"./results/{args.collection}/{mech}/obfuscatedText_{mech}_{eps}.csv")
                #partition df into 5 parts for efficiency
                df = np.array_split(df, 5)
                for i in range(5):
                    data = pd.DataFrame(df[i].reset_index(), columns = df[0].columns)
                    attack.run(df = data, i = i)
        if mech == 'AEA':
            for level in levels:
                df = pd.read_csv(f"./results/{args.collection}/{mech}/obfuscatedText_{mech}_{level}.csv")
                df = np.array_split(df, 5)
                for i in range(5):
                    data = pd.DataFrame(df[i].reset_index(), columns = df[0].columns)
                    attack.run(df = data, i = i)
        if mech == 'FEA':
            for wins in winsize:
                df = pd.read_csv(f"./results/{args.collection}/{mech}/obfuscatedText_{mech}_{wins}.csv")
                df = np.array_split(df, 5)
                for i in range(5):
                    data = pd.DataFrame(df[i].reset_index(), columns = df[0].columns)
                    attack.run(df = data, i = i)

    os.makedirs('./QuIPU/logs/', exist_ok=True)
    os.system(f"cp ./logs/logger_{args.collection}.log ./QuIPU/logs/logger_{args.collection}.log")