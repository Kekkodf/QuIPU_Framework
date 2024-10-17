import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import time
from tqdm import tqdm
#import cosine similarity function
from sentence_transformers import SentenceTransformer
import numpy as np

tqdm.pandas()

class QuIPU():
    def __init__(self, 
                 logger: object, 
                 query_logs: str,
                 query_logs_embeddings: str,
                 model: object,
                 collection: str) -> None:
        '''
        QuIPU object constructor
        '''
        self.logger: object = logger
        # Load query logs embeddings directly as a numpy array.
        self.query_logs_embeddings: np.array = np.load(query_logs_embeddings)['arr_0']
        # Load query logs from a CSV file.
        self.query_logs: pd.DataFrame = pd.read_csv(query_logs)
        self.collection: str = collection
        self.query_logs.drop(self.query_logs.columns[self.query_logs.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        self.logger.info(f"Query logs loaded. Shape of the embeddings: {self.query_logs_embeddings.shape}")
        self.model: object = model
        self.logger.info("SentenceTransformer model loaded and QuIPU object created.")
        self.logger.info(f"Collection: {self.collection}")

    def compute_similarity(self, 
                           x: np.array,
                           y: np.array) -> np.array:
        
        '''
        Compute the cosine similarity between two tensor like arrays

        :param x: tensor like array size (n, m)

        :param y: tensor like array size (k, m)

        :return: cosine similarity between x and y size (n, k)
        '''
        #cast the input arrays as two dimensional arrays
        #self.logger.info("Computing similarity")
        #self.logger.info(f"x shape: {x.shape}, y shape: {y.shape}")
        x_expanded: np.array = x[np.newaxis, :, :]
        y_expanded: np.array = y[:,np.newaxis, :]
        return np.sum(x_expanded * y_expanded, axis=2) / (np.linalg.norm(x_expanded, axis=2) * np.linalg.norm(y_expanded, axis=2))

    def encodeQueries(self, queries: list) -> np.array:
        '''
        Encode queries using the model and return their embeddings as a 2D numpy array.
        '''
        return self.model.encode(queries)

    def run(self, df: pd.DataFrame, i:int=1) -> None:
        '''
        Run method to simulate the Honest-but-Curious IR system.
        '''
        try:
            self.logger.info(f"QuIPU run started - Mechanism: {df['mechanism'][0]}, Epsilon: {df['epsilon'][0]}, index_partition: {i}")
            start = time.time()
        except:
            self.logger.info(f"QuIPU run started - Mechanism: {df['mechanism'][0]}, Level: {df['level'][0]}, index_partition: {i}")
            start = time.time()
        try:

            # Extract unique original texts and their IDs.
            original_texts = df[['id', 'text']].drop_duplicates().reset_index(drop=True)
            #sort the original texts by id
            original_texts = original_texts.sort_values(by='id').reset_index(drop=True)
            # Group by id and text to get the list of obfuscated texts.
            df_idText_ListObfuscatedText: pd.DataFrame = df.groupby(['id', 'text'])['obfuscatedText'].apply(list).reset_index()
            # Encode obfuscated texts.
            obfuscated_embeddings: pd.DataFrame = df_idText_ListObfuscatedText['obfuscatedText'].progress_apply(self.encodeQueries)
            self.logger.info("Obfuscated texts encoded.")
            # Compute the average obfuscated text embeddings.
            centroids: np.array = obfuscated_embeddings.apply(lambda x: np.mean(x, axis=0)).values
            #generate a matrix with the obfuscated embeddings
            centroids = np.array([x for x in centroids])
            
            self.logger.info("Centroid of the obfuscated text embeddings computed.")
            # Compute similarities between the obfuscated embeddings and query log embeddings.
            similarities = self.compute_similarity(centroids, self.query_logs_embeddings).T
            self.logger.info("Similarities between centroid and logs computed.")
            #self.logger.info(f"Similarities shape: {similarities.shape}")
            # Get the indices of the most similar query logs.
            sorted_indices = np.argsort(similarities, axis=1)[:, ::-1]            
            #self.logger.info(f"Similarities sorted. Shape: {sorted_indices.shape}")
            # Extract the most similar query log texts.
            #for each row in the similarities_sorted_indices, get the corresponding text from the query_logs dataframe
            #most_similar_query_logs: List[List[str]] = [[self.query_logs['text'][sorted_indices[i][j]] for j in range(len(sorted_indices[i]))] for i in range(len(sorted_indices))]
            most_similar_query_logs: List[List[str]] = [[self.query_logs['text'][i] for i in sorted_indices[j]] for j in range(len(sorted_indices))]
            self.logger.info("Most similar query logs extracted.")
            #now for each original text, add to the corresponding row the query logs that are most similar to it
            original_texts['most_similar_query_logs'] = [most_similar_query_logs[i] for i in range(len(original_texts))]

            #compute the P@1, R@10 and MRR
            P_at_1: List[float] = [
                                    1 if most_similar_query_logs[i][0] == original_texts['text'][i] else 0 for i in range(len(original_texts))
                                    ]
            #self.logger.info("P@1 computed.")

            R_at_10 = [
                        len(set([text.strip().lower() for text in most_similar_query_logs[i][:10]]).intersection(set([original_texts['text'][i].strip().lower()]))) / len(set([original_texts['text'][i].strip().lower()]))
                        if len(set([original_texts['text'][i].strip().lower()])) > 0 else 0
                        for i in range(len(original_texts))
                        ]
            #self.logger.info("R@10 computed.")

            MRR: List[float] = [
                                1/(1 + most_similar_query_logs[i].index(original_texts['text'][i])) if original_texts['text'][i] in most_similar_query_logs[i] else 0 for i in range(len(original_texts))
                                ]
            #self.logger.info("MRR computed.")

            original_texts['P@1'] = P_at_1
            original_texts['R@10'] = R_at_10
            original_texts['MRR'] = MRR

            #print the average P@1, R@10 and MRR
            self.logger.info(f"Average P@1: {np.mean(P_at_1)}")
            self.logger.info(f"Average R@10: {np.mean(R_at_10)}")
            self.logger.info(f"Average MRR: {np.mean(MRR)}")

            original_texts.drop('most_similar_query_logs', axis=1, inplace=True)
            print(original_texts.head())

            # Save the results to a CSV file.
            if df['mechanism'][0] in ['CMP', 'Mahalanobis', 'VickreyCMP', 'VickreyMhl', 'CusText', 'SanText', 'WBB', 'TEM']:
                os.makedirs(f'./QuIPU/results/{self.collection}/{df['mechanism'][0]}', exist_ok=True)
                original_texts.to_csv(f"./QuIPU/results/{self.collection}/{df['mechanism'][0]}/QuIPU_{df['mechanism'][0]}_{df['epsilon'][0]}_{i}.csv", index=False)
            if df['mechanism'][0] in ['AEA']:
                os.makedirs(f'./QuIPU/results/{self.collection}/{df['mechanism'][0]}', exist_ok=True)
                original_texts.to_csv(f"./QuIPU/results/{self.collection}/{df['mechanism'][0]}/QuIPU_{df['mechanism'][0]}_{df['level'][0]}_{i}.csv", index=False)
            if df['mechanism'][0] in ['FEA']:
                os.makedirs(f'./QuIPU/results/{self.collection}/{df['mechanism'][0]}', exist_ok=True)
                original_texts.to_csv(f"./QuIPU/results/{self.collection}/{df['mechanism'][0]}/QuIPU_{df['mechanism'][0]}_{df['level'][0]}_{i}.csv", index=False)
            
            end = time.time()
            self.logger.info(f"QuIPU run finished. Time elapsed: {end - start} seconds")

        except Exception as e:
            self.logger.error(f"Error in QuIPU run: {str(e)}")
            return None
