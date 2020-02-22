from .q_learner import QLearner
from .coma_learner import COMALearner
from .categorical_q_learner import CateQLearner


REGISTRY = {
    "q_learner": QLearner,
    "coma_learner": COMALearner,
    "cate_q_learner": CateQLearner
}
