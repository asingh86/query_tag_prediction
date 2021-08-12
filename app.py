from model.bert import BertModel
from transformer import Transformer
from utils import metrics


def model_fit_predict():
    t = Transformer()
    b = BertModel()
    m = metrics

    x_train, x_val, x_test, y_train, y_val, y_test, target_label = t.train_val_test_split()

    multiclass_model, history = b.model_fit(x_train, x_val, y_train, y_val)

    y_pred = b.perform_inference(x_test)
    accuracy = metrics.categorical_accuracy(y_test, y_pred)

    return accuracy
