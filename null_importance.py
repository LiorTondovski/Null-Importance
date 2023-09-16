import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple


class NullImportance():
    def __init__(self,
                 classifier:callable,
                 X:pd.DataFrame,
                 y:pd.Series,
                 num_permutations:int,
                 num_real_feature_importance_calculations:int,
                 p_value:float):
        
        self.classifier = classifier
        self.X = X
        self.y = y
        self.num_permutations = num_permutations
        self.num_real_feature_importance_calculations = num_real_feature_importance_calculations
        self.p_value = p_value


    def target_permutation(self) -> List[np.array]:
        """
        Create the target permutations 
            Parameters
            ----------
            Returns
            ------
            target_permutations: List[np.array]
        """
        target_permutations = []
        for i in range(self.num_permutations):
            target_permutations.append(np.random.permutation(self.y))
        return target_permutations


    def get_null_importances(self, target_permutaions:List[np.array]) -> Dict[str, List[float]]:
        """
        Calculate the null importances values for each feature according to the target permutations.
            Parameters
            ----------
            target_permutaions: List[np.array]

            Returns
            ------
            null_importance: Dictionary[str, List[float]]
        """
        null_importance = {}

        for i in tqdm(range(self.num_permutations)):
            classifier = self.classifier
            classifier.fit(self.X, target_permutaions[i])
            feature_iportance = classifier.feature_importances_

            for j, column in enumerate(list(self.X.columns)):
                null_importance.setdefault(column,[]).append(feature_iportance[j])

        return null_importance


    def get_actual_importance(self) -> Dict[str, List[float]]:
        """
        Calculate the importances values for each feature according to the real target.
            Parameters
            ---------- 
            Returns
            ------
        actual_importance : Dictionary
        """
        actual_importance = {}

        for i in range(self.num_real_feature_importance_calculations):
            classifier = self.classifier
            classifier.fit(self.X, self.y)
            feature_iportance = classifier.feature_importances_

            for j, column in enumerate(list(self.X.columns)):
                actual_importance.setdefault(column,[]).append(feature_iportance[j])
        
        for key in actual_importance.keys():
            actual_importance[key] = np.mean(actual_importance[key])

        return actual_importance
    

    def feature_selection(self,
                          null_importances:Dict[str, List[float]],
                          actual_importances:Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        Feature selection step
            
            Parameters
            ----------
            null_importances : Dict[str, List[float]
            actual_importances : Dict[str, float]
            
            Returns
            ------
            eliminated_features : List
            selected_features : List
        """

        eliminated_features = []
        selected_features = []

        for feature in list(actual_importances.keys()):
            num_importace_scores = self.num_permutations
            scores_smaller_than_actual = 0
            
            for sample in null_importances[feature]:
                if sample <= actual_importances[feature]:
                    scores_smaller_than_actual+=1

            ratio = scores_smaller_than_actual / num_importace_scores

            if ratio >= 1 - self.p_value:
                selected_features.append(feature)
            else:
                eliminated_features.append(feature)

        return eliminated_features, selected_features
    

    def print_results(self, eliminated_features:List[str], selected_features:List[str])->None:
        """
        Feature selection summary
            
            Parameters
            ---------- 
            eliminated_features : List[str]
            selected_features : List[str]
            Returns
            ------
        """
        num_features = len(eliminated_features + selected_features)
        eliminated_ratio = round((len(eliminated_features)/num_features)*100, 2)
        print(f'The total number of features before feature selection is: {num_features}')
        print(f'The total number of eliminated features is {len(eliminated_features)}')
        print(f'{eliminated_ratio}% of the original features were eliminated')
        print(f'The total number of selected features is {len(selected_features)}')


    def feature_selection_pipline(self) -> Tuple[List[str], List[str]]:
        """
        Runs null feature importance feature selection pipline
            
            Parameters
            ---------- 
            Returns
            ------
            eliminated_features : List
            selected_features : List
        """
        target_permutations = self.target_permutation()
        null_importances = self.get_null_importances(target_permutations)
        actual_importance = self.get_actual_importance()
        eliminated_features, selected_features = self.feature_selection(null_importances, actual_importance)
        self.print_results(eliminated_features, selected_features)
        return selected_features, eliminated_features
        