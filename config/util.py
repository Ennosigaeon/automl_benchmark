from autosklearn.pipeline.components.base import AutoSklearnComponent
from autosklearn.pipeline.components.classification.adaboost import AdaboostClassifier
from autosklearn.pipeline.components.classification.bernoulli_nb import BernoulliNB
from autosklearn.pipeline.components.classification.decision_tree import DecisionTree
from autosklearn.pipeline.components.classification.extra_trees import ExtraTreesClassifier
from autosklearn.pipeline.components.classification.gaussian_nb import GaussianNB
from autosklearn.pipeline.components.classification.gradient_boosting import GradientBoostingClassifier
from autosklearn.pipeline.components.classification.k_nearest_neighbors import KNearestNeighborsClassifier
from autosklearn.pipeline.components.classification.lda import LDA
from autosklearn.pipeline.components.classification.liblinear_svc import LibLinear_SVC
from autosklearn.pipeline.components.classification.libsvm_svc import LibSVM_SVC
from autosklearn.pipeline.components.classification.multinomial_nb import MultinomialNB
from autosklearn.pipeline.components.classification.passive_aggressive import PassiveAggressive
from autosklearn.pipeline.components.classification.qda import QDA
from autosklearn.pipeline.components.classification.random_forest import RandomForest
from autosklearn.pipeline.components.classification.sgd import SGD


class ConfigSpace:

    @staticmethod
    def sklearn_mapping(sklearn: str) -> type(AutoSklearnComponent):
        if sklearn == '':
            return AdaboostClassifier
        elif sklearn == 'sklearn.naive_bayes.BernoulliNB':
            return BernoulliNB
        elif sklearn == 'sklearn.tree.DecisionTreeClassifier':
            return DecisionTree
        elif sklearn == 'sklearn.ensemble.ExtraTreesClassifier':
            return ExtraTreesClassifier
        elif sklearn == 'sklearn.naive_bayes.GaussianNB':
            return GaussianNB
        elif sklearn == 'sklearn.ensemble.GradientBoostingClassifier':
            return GradientBoostingClassifier
        elif sklearn == 'sklearn.neighbors.KNeighborsClassifier':
            return KNearestNeighborsClassifier
        elif sklearn == 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis':
            return LDA
        elif sklearn == 'sklearn.svm.LinearSVC':
            return LibLinear_SVC
        elif sklearn == 'sklearn.svm.SVC':
            return LibSVM_SVC
        elif sklearn == 'sklearn.naive_bayes.MultinomialNB':
            return MultinomialNB
        elif sklearn == 'sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier':
            return PassiveAggressive
        elif sklearn == 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis':
            return QDA
        elif sklearn == 'sklearn.ensemble.RandomForestClassifier':
            return RandomForest
        elif sklearn == 'sklearn.linear_model.stochastic_gradient.SGDClassifier':
            return SGD
        # elif sklearn == '':
        #     return XGradientBoostingClassifier
        raise NotImplementedError('Algorithm {} is not implemented yet'.format(sklearn))
