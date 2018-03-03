import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter concepts or entities from predictions')
    parser.add_argument(
        '--predictions',
        default='/mnt/store/friend/proj/SemEval18-hypernym/results/test/predictions/1A.english_1A_UMBC_tokenized.txt_100_sg.vec.gz_True_200_0.3_unit_True_vocabulary_filtered.alph.reduced2_more_permissive.dot_True.1.0.True.output.txt')
    # az 1.0 baloldalán álló az az, hogy az attpárok be voltak-e kapcsolva
    # a jobboldali pedig az h filterezve volt-e a candidate hipernímák halmaza
    # ez után az érték fog majd a következő verzióba egy olyan egész is
    # bekerülni, ami pedig a tanítás során látott negatív példák száma
    return parser.parse_args()


def main():
    args = parse_args()
    pred_dir, pred_basefn = os.path.split(args.predictions)
    subtask = pred_basefn.split('_')[0]
    assert ('dev' in args.predictions) != ('test' in args.predictions)
    phase = 'trial' if 'dev' in args.predictions else 'test'
    proj_dir = '/mnt/store/friend/proj/SemEval18-hypernym/'

    def input_fn(data_or_gold):
        input_dir = os.path.join(
            proj_dir, 'SemEval18-Task9/{}/{}'.format(phase, data_or_gold))
        input_basefn = '{}.{}.{}.txt'.format(subtask, phase, data_or_gold)
        return input_dir, input_basefn
    query_dir, query_basefn = input_fn('data')
    gold_dir, gold_basefn = input_fn('gold')
    for categ in ['concept', 'entity']:
        for all_dir in query_dir, gold_dir:
            categ_dir = os.path.join(all_dir, categ)
            if not os.path.isdir(categ_dir):
                os.mkdir(categ_dir)
        categ_dir = os.path.join(pred_dir, categ)
        if not os.path.isdir(categ_dir):
                os.mkdir(categ_dir)
    query_path = os.path.join(query_dir, query_basefn)
    gold_path = os.path.join(gold_dir, gold_basefn)
    gold_entity_path = os.path.join(gold_dir, 'entity', gold_basefn)
    gold_concept_path = os.path.join(gold_dir, 'concept', gold_basefn)
    pred_entity_path = os.path.join(pred_dir, 'entity', pred_basefn)
    pred_concept_path = os.path.join(pred_dir, 'concept', pred_basefn)
    with open(query_path) as query_f, open(gold_path) as gold_f, \
            open(args.predictions) as pred_f, \
            open(gold_entity_path, mode='w') as gold_entity_f, \
            open(gold_concept_path, mode='w') as gold_concept_f, \
            open(pred_entity_path, mode='w') as pred_entity_f, \
            open(pred_concept_path, mode='w') as pred_concept_f:
        categ = query_f.readline().strip().split('\t')[1]
        gold_l = gold_f.readline()
        pred_l = pred_f.readline()
        if categ == 'Entity':
            gold_entity_f.write(gold_l)
            pred_entity_f.write(pred_l)
        else:
            gold_concept_f.write(gold_l)
            pred_concept_f.write(pred_l)


if __name__ == '__main__':
    main()
