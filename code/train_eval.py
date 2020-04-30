from paras import load_params
from dataset import read_tweets, get_voca
from embed import *
from evaluator import QuantitativeEvaluator, QualitativeEvaluator
from valid_test import CrossTestData
import dill as pickle
import types

def set_rand_seed(pd):
    rand_seed = pd['rand_seed']
    np.random.seed(rand_seed)
    random.seed(rand_seed)


def read_data(pd):
    start_time = time.time()
    tweets = read_tweets(pd['tweet_file'])
    random.shuffle(tweets)
    print("Total number of tweets: {}".format(len(tweets)))
    train_test = pd['test_size'] + pd['train_size'] + pd['valid_size']
    tweets = tweets[:train_test]
    voca = get_voca(tweets, pd['voca_min'], pd['voca_max'])
    train_data = tweets[:pd['train_size']]
    valid_data = tweets[pd['train_size']:pd['train_size']+pd['valid_size']]
    test_data = tweets[pd['train_size']+pd['valid_size']:]
    print('Reading data done, elapsed time: {}'.format(round(time.time()-start_time)))
    print('Number of training tweets: {}'.format(len(train_data))) 
    print('Number of valid tweets: {}'.format(len(valid_data))) 
    print('Number of testing tweets: {}'.format(len(test_data))) 
    return train_data, valid_data, test_data, voca


def train_model(train_data, voca):
    start_time = time.time()
    predictor = CrossMap(pd)
    predictor.fit(train_data, voca)   
    print('Model training done, elapsed time: {}'.format(round(time.time()-start_time)))
    return predictor


def predict(model, test_data, pd):
    start_time = time.time()
    for t in pd['predict_type']:
        evaluator = QuantitativeEvaluator(predict_type=t)
        if pd['new_test_method']:
            test_graph = CrossTestData(pd)
            evaluator.get_ranks_from_test_graph(test_data, model, test_graph)
            mrr, mr = evaluator.compute_highest_mrr()
            print('Type:{} hmr: {}, hmrr: {} '.format(evaluator.predict_type, mr, mrr))
        else:
            evaluator.get_ranks(test_data, model)
            mrr, mr = evaluator.compute_mrr()
            print('Type:{} mr: {}, mrr: {} '.format(evaluator.predict_type, mr, mrr))
    print('Prediction done. Elapsed time: {}'.format(round(time.time()-start_time))) 


def write_embeddings(model, pd):
    directory = pd['model_embeddings_dir']
    if not os.path.isdir(directory):
        os.makedirs(directory)
    for nt, vecs in model.nt2vecs.items():
        with open(directory+nt+'.txt', 'w') as f:
            for node, vec in vecs.items():
                if nt=='l':
                    node = model.lClus.centroids[node]
                if nt=='t':
                    node = model.tClus.centroids[node]
                l = [str(e) for e in [node, list(vec)]]
                f.write('\x01'.join(l)+'\n')


def write_numbered_embed(model, pd):
    embedding_file = pd['model_embeddings_dir']
    # if not os.path.isdir(directory):
    #     os.makedirs(directory)
    embed_f = open(embedding_file, 'w')
    for nt, vecs in model.nt2vecs.items():
        for node, vec in vecs.items():
            if nt=='l':
                node = model.lClus.centroids[node]
            if nt=='t':
                node = model.tClus.centroids[node]
            try:
                node_num = model.node_id[str(node)]
            except:
                print(node)
            embed_l = [str(e) for e in [node_num]+list(vec)]
            embed_f.write('\t'.join(embed_l)+'\n')
    embed_f.close()


def run_case_study(model, pd):
    start_time = time.time()
    evaluator = QualitativeEvaluator(model, pd['case_dir'])
    for word in ['food', 'restaurant', 'beach', 'weather', 'clothes', 'nba']:
        evaluator.getNbs1(word)
    for location in [[34.043021,-118.2690243], [33.9424, -118.4137], [34.008, -118.4961], [34.0711, -118.4434]]:
        evaluator.getNbs1(location)
    evaluator.getNbs2('outdoor', 'weekend')
    print 'Case study done. Elapsed time: ', round(time.time()-start_time)


def run(pd):
    set_rand_seed(pd)
    train_data, valid_data, test_data, voca = read_data(pd)
    pickle.dump(train_data, open(pd['train_data_path'],'w'))
    pickle.dump(valid_data, open(pd['valid_data_path'],'w'))    
    pickle.dump(test_data, open(pd['test_data_path'],'w'))
    load_existing_model = pd['load_existing_model']
    if load_existing_model:
        model = pickle.load(open(pd['model_pickled_path'],'r'))
        write_numbered_embed(model, pd)        
    else:        
        model = train_model(train_data, voca)
        # write_embeddings(model, pd)
        write_numbered_embed(model, pd)
        pickle.dump(model, open(pd['model_pickled_path'],'w'))
    predict(model, test_data, pd)
    perform_case_study = pd['perform_case_study']
    if perform_case_study:
        run_case_study(model, pd)


if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    run(pd)
