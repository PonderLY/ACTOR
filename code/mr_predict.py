from paras import load_params
from dataset import read_tweets, get_voca
from embed import *
from evaluator import QuantitativeEvaluator, QualitativeEvaluator
from crossdata import CrossData
import dill as pickle
import types

def set_rand_seed(pd):
    rand_seed = pd['rand_seed']
    np.random.seed(rand_seed)
    random.seed(rand_seed)


def predict(model, test_data, pd):
    start_time = time.time()
    test_graph = CrossData(pd['node_dict'], pd['test_edges']) 
    for t in pd['predict_type']:
        evaluator = QuantitativeEvaluator(predict_type=t)
        if pd['new_test_method']:
            evaluator.get_ranks_from_test_graph(test_data, model, test_graph)
            mrr, mr = evaluator.compute_highest_mrr()
            print('Type:{} hmr: {}, hmrr: {} '.format(evaluator.predict_type, mr, mrr))
        else:
            evaluator.get_ranks(test_data, model)
            mrr, mr = evaluator.compute_mrr()
            print('Type:{} mr: {}, mrr: {} '.format(evaluator.predict_type, mr, mrr))
    print('Prediction done. Elapsed time: {}'.format(round(time.time()-start_time))) 


def run_case_study(model, pd):
    start_time = time.time()
    evaluator = QualitativeEvaluator(model, pd['case_dir'])
    for word in ['food', 'restaurant', 'beach', 'weather', 'clothes', 'nba']:
        evaluator.getNbs1(word)
    for location in [[34.043021,-118.2690243], [33.9424, -118.4137], [34.008, -118.4961], [34.0711, -118.4434]]:
        evaluator.getNbs1(location)
    evaluator.getNbs2('outdoor', 'weekend')
    print('Case study done. Elapsed time:{} '.format(round(time.time()-start_time)))


def run(pd):
    set_rand_seed(pd)
    start_time = time.time()
    test_data = pickle.load(open(pd['test_data_path'],'r'))
    predictor = pickle.load(open(pd['model_pickled_path'],'r'))
    print('Model and test data read done. Elapsed time:{} '.format(round(time.time()-start_time)))
    # predictor.read_embedding(pd)
    # print 'Graph embedding read done, elapsed time: ', round(time.time()-start_time)
    predict(predictor, test_data, pd)
    if pd['perform_case_study']:
        run_case_study(predictor, pd)


if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    run(pd)
