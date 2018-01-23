import json
import argparse
from collections import Counter


def generate_file_names(args, mapping):
    data_file = '{}/training/data/{}.{}.training.data.txt'.format(
        args.dataset_dir,
        args.dataset_id,
        mapping[args.dataset_id][0]
    )
    gold_file = '{}/training/gold/{}.{}.training.gold.txt'.format(
        args.dataset_dir,
        args.dataset_id,
        mapping[args.dataset_id][0]
    )
    vocab_file='{}/vocabulary/{}.{}.vocabulary.txt'.format(
        args.dataset_dir,
        args.dataset_id,
        mapping[args.dataset_id][0]
    )
    return data_file, gold_file, vocab_file


def find_paths_to_object(J, query):
    '''
    All the occurrences of the query object is searched for within the concept tree J created with In-close.
    Paths to the within-tree occurrence of the query string are returned as a list of nodes towards less specific nodes (eventually all of them ending in the root of the tree.

    usage example:

    J = json.load(open('1A_UMBC_tokenized.txt_100_cbow.vec.gz_True_1000_0.2_unit_True_vocabulary_filtered_reduced.cxt.json')
    paths, nodes, depths = find_paths_to_object(J, 'tropical_storm')
    '''
    to_visit = [J]
    own_nodes = []
    depths = {J['Node']:0}
    parents = {J['Node']:-1} # the root has no parent
    previous_nodes = {0: J}
    depths_to_return = []
    paths = []
    visited_nodes = {}
    while len(to_visit) > 0:
        visit_next = to_visit.pop(0)
        if 'children' in visit_next:
            for child in visit_next['children']:
                depths[child['Node']] = depths[visit_next['Node']] + 1
                parents[child['Node']] = visit_next['Node']
                previous_nodes[child['Node']] = child
                if query in child['own_objects']:
                    #print(child)
                    parent = parents[child['Node']]
                    path = []
                    to_remove = set(child['own_objects'])
                    while parent != -1:
                        pn = previous_nodes[parent].copy()
                        del pn['children'] # for better readibility
                        del pn['ObjectCount']
                        del pn['objects']
                        path.append(pn)
                        parent = parents[parent]
                    paths.append(path)
                    own_nodes.append(child['own_objects'])
                    depths_to_return.append(depths[child['Node']])
                elif query in child['objects']:
                    to_visit.append(child)
    return paths, own_nodes, depths_to_return


def main():
    dataset_mapping={
        '1A':['english', 'UMBC'],
        '1B':['italian', 'it_itwac'],
        '1C':['spanish', 'es_1Billion'],
        '2A':['medical', 'med_pubmed'],
        '2B':['music', 'music_bioreviews']
    }

    parser = argparse.ArgumentParser(description='SemEval2018 Task9 FCA script')
    parser.add_argument('--dataset_id', required=False, default='1A', choices=dataset_mapping.keys())
    parser.add_argument('--query_word', required=False, default='tropical_storm')
    parser.add_argument('--input_embedding', default='sg', choices=['sg', 'cbow'])
    parser.add_argument('--regularizer', default=0.3, choices=[0.2, 0.3, 0.4, 0.5], type=float)
    parser.add_argument('--fca_dir', default='/home/berend/datasets/semeval2018/task9_FCA_data')
    parser.add_argument('--dataset_dir', default='/home/berend/datasets/semeval2018/SemEval18-Task9')
    args = parser.parse_args()

    f='{}/{}_{}_tokenized.txt_100_{}.vec.gz_True_1000_{}_unit_True_vocabulary_filtered_reduced.cxt.json'.format(
        args.fca_dir,
        args.dataset_id,
        dataset_mapping[args.dataset_id][1],
        args.input_embedding,
        args.regularizer
    )
    print(f)
    J = json.load(open(f))

    parents, own_nodes, depths_to_return = find_paths_to_object(J, args.query_word)
    common_neighbors = Counter([o for p in own_nodes for o in p])
    for x in common_neighbors.most_common(15):
        print(x)

    '''
    w2i = set(J['objects'])

    data_file, gold_file, vocab_file = generate_file_names(args, dataset_mapping)
    possible_hypernyms = [l.strip().replace(' ', '_')
                          for l in open(vocab_file, 'r') if l.strip().replace(' ', '_') in w2i and len(l.strip()) > 1]
    train_queries = [l.split('\t')[0].replace(' ', '_')
                     for l in open(data_file) if l.split('\t')[0].replace(' ', '_') in w2i]
    train_line_included = [l.split('\t')[0].replace(' ', '_') in w2i
                           for l in open(data_file)] # is the line OOV
    train_golds = [
        [x.replace(' ', '_') for x in line.strip().split('\t') if x.replace(' ', '_') in w2i]
        for i,line in enumerate(open(gold_file)) if train_line_included[i]
    ]

    dev_queries = [l.split('\t')[0].replace(' ', '_')
                   for l in open(data_file.replace('training', 'trial')) if l.split('\t')[0].replace(' ', '_') in w2i]
    dev_line_included = [l.split('\t')[0].replace(' ', '_') in w2i
                         for l in open(data_file.replace('training', 'trial'))] # is the line OOV
    dev_golds=[
        [x.replace(' ', '_') for x in line.strip().split('\t') if x.replace(' ', '_') in w2i]
        for i,line in enumerate(open(gold_file.replace('training', 'trial'))) if dev_line_included[i]
    ]
    test_queries=[l.split('\t')[0].replace(' ', '_') for l in open(data_file.replace('training', 'test'))]

    for tq in test_queries:
        if tq in w2i:
            pass # this is the best which could happen
        elif tq.split('_')[0] in w2i:
            print(tq, tq.split('_')[0].lower())
        elif tq.split('_')[0].lower() in w2i:
            print(tq, tq.split('_')[0].lower())
        elif tq.lower() in w2i:
            print(tq, tq.lower())
        else:
            print(tq, 'Hopeless')
    '''


if __name__ == "__main__":
    main()
