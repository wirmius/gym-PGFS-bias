from train_mgenfail_scorer import train_rfc_mgenfail

from sys import argv

assay = argv[1]
n_folds = int(argv[2])

train_rfc_mgenfail(True,
                   dataset=assay,
                   hosts={'server': '127.0.0.1', 'port': 5555},
                   n_tries=n_folds
                   )