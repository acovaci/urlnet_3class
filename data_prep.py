import dask.dataframe as dd
from distributed import Client

from loguru import logger

logger.add(f'.logs/data_prep.txt', rotation="500 MB")

def print_stats(train, test=None):
    obs_train = len(train)
    if test is not None:
        obs_test = len(test)
    phis_train = len(train[train['phishing'] == '+1'])
    if test is not None:
        phis_test = len(test[test['phishing'] == '+1'])

    logger.debug(f'Train entries: {obs_train}')
    if test is not None:
        logger.debug(f'Test entries: {obs_train}')
    logger.debug(f'Train phishing: {phis_train} ({phis_train / obs_train})')
    if test is not None:
        logger.debug(f'Test phishing: {phis_test} ({phis_test / obs_test})')

if __name__ == '__main__':
    c = Client(n_workers=12)

    data = dd.read_csv('/mnt/data/csv_dns_testing_7.csv/*')

    data['phishing'] = data['phishing_aph'].apply(
        lambda x: '-1' if x == 'clean' else '+1' if x == 'generated' else '+2')

    data = data[['phishing', 'fqdn']]

    data_train, data_test = data.random_split([0.5, 0.5])

    for frac in [0.0001]:
        if frac == 1:
            data_train.to_csv('/mnt/data/strong_dns_augmented/urlnet/urlnet-train-full.txt',
                            sep='\t', header=None, single_file=True, index=False)
            logger.info(f'[{frac}] Output /mnt/data/strong_dns_augmented/urlnet/urlnet-train-full.txt')
            data_test.to_csv('/mnt/data/strong_dns_augmented/urlnet/urlnet-test-full.txt',
                            sep='\t', header=None, single_file=True, index=False)
            logger.info(f'[{frac}] Output /mnt/data/strong_dns_augmented/urlnet/urlnet-test-full.txt')
            print_stats(data_train, data_test)
        else:
            data_train_ = data_train.sample(frac=0.02)
            data_test_ = data_test.sample(frac=0.02)
            data_train_.to_csv(f'/mnt/data/strong_dns_augmented/urlnet/urlnet-train-{5e7*frac}.txt',
                            sep='\t', header=None, single_file=True, index=False)
            logger.info(f'[{frac}] Output /mnt/data/strong_dns_augmented/urlnet/urlnet-train-{5e7*frac}.txt')
            data_test_.to_csv(f'/mnt/data/strong_dns_augmented/urlnet/urlnet-test-{5e7*frac}.txt',
                            sep='\t', header=None, single_file=True, index=False)
            logger.info(f'[{frac}] Output /mnt/data/strong_dns_augmented/urlnet/urlnet-test-{5e7*frac}.txt')
            print_stats(data_train_, data_test_)

# 2020-01-06 10: 52: 24.613 | DEBUG | __main__: print_stats: 14 - Train entries: 54193150
# 2020-01-06 10: 52: 24.613 | DEBUG | __main__: print_stats: 16 - Test entries: 54193150
# 2020-01-06 10: 52: 24.613 | DEBUG | __main__: print_stats: 17 - Train phishing: 1412461 (0.026063460049840247)
# 2020-01-06 10: 52: 24.614 | DEBUG | __main__: print_stats: 19 - Test phishing: 1408307 (0.025995002553433788)
