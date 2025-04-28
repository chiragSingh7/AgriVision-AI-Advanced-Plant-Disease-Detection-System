"""
Machine Learning classifiers for plant disease classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path
from .config import ML_CONFIG

# Get logger instead of using basicConfig
logger = logging.getLogger(__name__)

class MLClassifiers:
    def __init__(self, random_state=None):
        """Initialize ML classifiers."""
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state or ML_CONFIG["random_forest"]["random_state"]
        
        # Initialize classifiers
        self.rf_clf = RandomForestClassifier(
            n_estimators=ML_CONFIG["random_forest"]["n_estimators"],
            max_depth=ML_CONFIG["random_forest"]["max_depth"],
            random_state=self.random_state
        )
        
        self.svm_clf = SVC(
            kernel=ML_CONFIG["svm"]["kernel"],
            C=ML_CONFIG["svm"]["C"],
            probability=ML_CONFIG["svm"]["probability"],
            random_state=self.random_state
        )
        
        self.lr_clf = LogisticRegression(
            multi_class=ML_CONFIG["logistic_regression"]["multi_class"],
            max_iter=ML_CONFIG["logistic_regression"]["max_iter"],
            random_state=self.random_state
        )
        
        # Initialize voting classifier
        self.voting_clf = VotingClassifier(
            estimators=[
                ('rf', self.rf_clf),
                ('svm', self.svm_clf),
                ('lr', self.lr_clf)
            ],
            voting=ML_CONFIG["voting"]["voting"],
            weights=ML_CONFIG["voting"]["weights"]
        )
        
        self.logger.info("ML classifiers initialized")
    
    def train_random_forest(self, X, y):
        """Train Random Forest classifier."""
        try:
            self.rf_clf.fit(X, y)
            self.logger.info("Random Forest training completed")
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {str(e)}")
            raise
    
    def train_svm(self, X, y):
        """Train SVM classifier."""
        try:
            self.svm_clf.fit(X, y)
            self.logger.info("SVM training completed")
        except Exception as e:
            self.logger.error(f"Error training SVM: {str(e)}")
            raise
    
    def train_logistic_regression(self, X, y):
        """Train Logistic Regression classifier."""
        try:
            self.lr_clf.fit(X, y)
            self.logger.info("Logistic Regression training completed")
        except Exception as e:
            self.logger.error(f"Error training Logistic Regression: {str(e)}")
            raise
    
    def train_voting_classifier(self, X, y):
        """Train Voting classifier."""
        try:
            self.voting_clf.fit(X, y)
            self.logger.info("Voting Classifier training completed")
        except Exception as e:
            self.logger.error(f"Error training Voting Classifier: {str(e)}")
            raise
    
    def evaluate(self, X, y, class_names=None):
        """Evaluate all classifiers."""
        try:
            results = {}
            classifiers = {
                'Random Forest': self.rf_clf,
                'SVM': self.svm_clf,
                'Logistic Regression': self.lr_clf,
                'Voting Classifier': self.voting_clf
            }
            
            for name, clf in classifiers.items():
                y_pred = clf.predict(X)
                results[name] = {
                    'classification_report': classification_report(
                        y, y_pred, target_names=class_names, output_dict=True
                    ),
                    'confusion_matrix': confusion_matrix(y, y_pred)
                }
                self.logger.info(f"{name} evaluation completed")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error evaluating classifiers: {str(e)}")
            raise 